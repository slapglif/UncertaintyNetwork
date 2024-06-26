# .\core\models\uncertainty\uncertain_nn.py
from dataclasses import dataclass
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from core.models.embedding import RotaryPositionEncoding, apply_rotary_pos_emb
from core.models.layers import TransformerEncoderLayer, CEMA, KANFeedForward
from core.models.statespace import Mamba, MambaConfig
from core.models.uncertainty.uncertainty_utils import UncertaintyModule
from core.utils.tokenizer import Tokenizer
from core.utils.utils import TimestepNorm


class UncertainTransformerConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size: int = 50257,  # Set to match GPT-2's vocabulary size
            d_model: int = 768,
            n_heads: int = 12,
            d_ff: int = 3072,
            n_layers: int = 12,
            dropout: float = 0.1,
            max_position_embeddings: int = 1024,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = False,
            pad_token_id: int = 50256,  # Set to match GPT-2's pad token ID
            bos_token_id: int = 50256,  # Set to match GPT-2's BOS token ID
            eos_token_id: int = 50256,  # Set to match GPT-2's EOS token ID
            use_mamba: bool = True,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: float = 2,
            dt_rank: Optional[int] = None,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            use_flash_attention: bool = False,
            n_inducing: int = 10,
            use_gelu_approximation: bool = False,
            sliding_window_size: int = 512,
            device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.use_mamba = use_mamba
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank if dt_rank is not None else self.d_model
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.use_flash_attention = use_flash_attention
        self.n_inducing = n_inducing
        self.use_gelu_approximation = use_gelu_approximation
        self.sliding_window_size = sliding_window_size
        self.device = device


class UncertainNN(nn.Module):
    def __init__(self, config: UncertainTransformerConfig):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_pos_emb = RotaryPositionEncoding(
            config.d_model, config.n_heads, config.max_position_embeddings
        )
        self.sentence_transformer = SentenceTransformer(
            "tomaarsen/mpnet-base-nli-matryoshka", device="cuda"
        )
        self.sentence_proj = nn.Linear(768, config.d_model).to(config.device)
        self.cema = CEMA(config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Mamba(
                            MambaConfig(
                                d_model=config.d_model,
                                d_state=config.d_state,
                                expand_factor=config.expand_factor,
                                d_conv=config.d_conv,
                                dt_min=config.dt_min,
                                dt_max=config.dt_max,
                                dt_init=config.dt_init,
                                dt_scale=config.dt_scale,
                                dt_init_floor=config.dt_init_floor,
                            )
                        ),
                        TransformerEncoderLayer(config),
                        KANFeedForward(config),
                        nn.Linear(config.d_model, config.d_model).to(config.device),
                    ]
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.uncertainty_module = UncertaintyModule(
            input_dim=config.d_model,
            output_dim=config.vocab_size,
            n_gp_layers=2,
            n_inducing=config.n_inducing,
            dropout_rate=config.dropout,
        ).to(config.device)
        self.use_cache = False # config.use_cache  # Add use_cache attribute
        self.dropout = nn.Dropout(config.dropout)  # Add dropout layer

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: bool = False,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        """
        Forward pass of the UncertainNN module.

        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (Optional[torch.FloatTensor]): Attention mask of shape (batch_size, seq_len).
            use_cache (bool): Whether to use the past key/values attentions. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
                - hidden_states (torch.Tensor): Output hidden states of shape (batch_size, seq_len, d_model).
                - cache (Optional[Tuple[torch.Tensor, ...]]): Tuple of cached tensors if use_cache is True, else None.
                - uncertainty (torch.Tensor): Uncertainty of shape (batch_size, seq_len, vocab_size).
        """
        inputs_embeds = self.embedding(input_ids)

        # Sentence embedding using SentenceTransformer
        with torch.no_grad():
            sentence_emb = self.sentence_transformer.encode(
                self.tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True),
                convert_to_tensor=True,
            )
        sentence_emb = self.sentence_proj(sentence_emb)
        sentence_emb = self.dropout(sentence_emb)  # Apply dropout after projection
        sentence_emb = (
            sentence_emb.unsqueeze(1)
            .repeat(1, inputs_embeds.size(1), 1)
            .to(inputs_embeds.device)
        )

        inputs_embeds = inputs_embeds + sentence_emb

        cos, sin = self.rotary_pos_emb(inputs_embeds, seq_len=input_ids.size(1))
        inputs_embeds = apply_rotary_pos_emb(inputs_embeds, cos, sin)

        inputs_embeds = self.cema(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds)  # Apply dropout after CEMA
        inputs_embeds = TimestepNorm(self.config.d_model)(inputs_embeds)

        hidden_states = inputs_embeds
        cache = () if use_cache else None  # Initialize cache if use_cache is True

        for mamba, transformer, kan_ff, projection in self.layers:
            mamba_outputs = mamba(hidden_states, use_cache=use_cache)
            hidden_states = mamba_outputs[0]  # Extract hidden states from Mamba output
            if use_cache:
                cache += (mamba_outputs[1],)  # Collect Mamba cache outputs

            hidden_states = projection(hidden_states)
            transformer_outputs = transformer(
                hidden_states, attention_mask=attention_mask, use_cache=use_cache
            )
            hidden_states = transformer_outputs[
                0
            ]  # Extract hidden states from Transformer output
            if use_cache:
                cache += transformer_outputs[1:]  # Collect Transformer cache outputs

            hidden_states = kan_ff(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        # No need to reshape here: hidden_states is (batch_size, input_dim)
        _, uncertainty = self.uncertainty_module(
            hidden_states
        )  # Extract uncertainty from UncertaintyModule output

        return hidden_states, cache, uncertainty


@dataclass
class UncertainCausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    uncertainty: Optional[torch.FloatTensor] = None


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config: UncertainTransformerConfig):
        super().__init__(config)
        self.config = config
        self.transformer = UncertainNN(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False).to(config.device)
        self.uncertainty_module = UncertaintyModule(
            input_dim=config.d_model,
            output_dim=config.vocab_size,
            n_gp_layers=2,
            n_inducing=config.n_inducing,
            dropout_rate=config.dropout,
        ).to(config.device)
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: bool = False,
            **kwargs: Any
    ) -> UncertainCausalLMOutputWithCrossAttentions:
        transformer_outputs = self.transformer(
            input_ids, attention_mask=attention_mask, use_cache=use_cache, **kwargs
        )

        hidden_states = transformer_outputs[0]
        cache = transformer_outputs[1]

        lm_logits = self.lm_head(hidden_states)

        # Calculate uncertainty
        _, uncertainty = self.uncertainty_module(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return UncertainCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=cache,
            hidden_states=hidden_states,
            attentions=None,
            cross_attentions=None,
            uncertainty=uncertainty,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding = value

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
