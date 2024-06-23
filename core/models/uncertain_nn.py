import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.embedding import PositionalEncoding, StableEmbedding
from core.models.layers import TransformerEncoderLayer


class UncertainTransformerConfig(PretrainedConfig):
    model_type = "uncertain_transformer"

    def __init__(
            self,
            vocab_size=50257,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_layers=6,
            dropout=0.1,
            max_position_embeddings=1024,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            pad_token_id=50256,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=True,
            use_stable_embedding=True,
            cema_hidden_dim=64,
            chunk_size=64,
            use_gelu_approximation=True,
            max_grad_norm=1.0,
            ntk_factor=1.0,
            uncertainty_factor=0.1,
            use_flash_attention=True,
            d_state=16,
            d_conv=4,
            mamba_expand_factor=2,
            mamba_dt_rank=None,
            mamba_dt_min=0.001,
            mamba_dt_max=0.1,
            **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
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
        self.use_stable_embedding = use_stable_embedding
        self.cema_hidden_dim = cema_hidden_dim
        self.chunk_size = chunk_size
        self.use_gelu_approximation = use_gelu_approximation
        self.max_grad_norm = max_grad_norm
        self.ntk_factor = ntk_factor
        self.uncertainty_factor = uncertainty_factor
        self.use_flash_attention = use_flash_attention
        self.d_state = d_state
        self.d_conv = d_conv
        self.mamba_expand_factor = mamba_expand_factor
        self.mamba_dt_rank = mamba_dt_rank
        self.mamba_dt_min = mamba_dt_min
        self.mamba_dt_max = mamba_dt_max

        # Add attribute map
        self.attribute_map = {"hidden_size": "d_model"}


class UncertainTransformer(nn.Module):
    def __init__(self, config: UncertainTransformerConfig | PretrainedConfig):
        super().__init__()
        self.config = config
        self.embedding = StableEmbedding(config.vocab_size, config.d_model,
                                         padding_idx=config.pad_token_id) if config.use_stable_embedding else nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_position_embeddings)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                output_attentions=False, output_hidden_states=False, return_dict=True):
        inputs_embeds = self.embedding(input_ids)
        hidden_states = self.pos_encoding(inputs_embeds)

        # Check for NaN/Inf after positional encoding
        self._check_nan_inf(hidden_states, "hidden_states after positional encoding")

        if self.gradient_checkpointing and self.training:
            hidden_states = checkpoint(self._forward_transformer, hidden_states, attention_mask, position_ids,
                                       past_key_values, use_cache, output_attentions, output_hidden_states)
        else:
            hidden_states = self._forward_transformer(hidden_states, attention_mask, position_ids, past_key_values,
                                                      use_cache, output_attentions, output_hidden_states)

        # Check for NaN/Inf after transformer layers
        self._check_nan_inf(hidden_states, "hidden_states after transformer layers")

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _forward_transformer(self, hidden_states, attention_mask, position_ids, past_key_values, use_cache,
                             output_attentions, output_hidden_states):
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            # Check for NaN/Inf after each layer
            self._check_nan_inf(hidden_states, f"hidden_states after layer {i}")

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _check_nan_inf(self, tensor: Tensor, message: str):
        """
        Checks if the given tensor contains NaN or Inf values and logs a warning if found.

        Args:
            tensor (torch.Tensor): The tensor to check.
            message (str): A message to include in the warning log.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN or Inf detected in {message}.")

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


class TemperatureSoftmaxLogitsProcessor(LogitsProcessor):
    """
    Processes the logits by applying a softmax with temperature scaling.

    Args:
        temperature (float): The temperature for scaling the logits before softmax.
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        """
        Applies temperature scaling and softmax to the input scores.

        Args:
            input_ids (torch.LongTensor): The input token IDs. Not used in this processor.
            scores (torch.FloatTensor): The logits to be processed.

        Returns:
            torch.Tensor: The processed logits after applying temperature scaling and softmax.
        """
        scores = scores / self.temperature

        # Numerically stable softmax calculation
        max_scores = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)
        probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
        return torch.log(probs)  # Return log probabilities


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = None
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.temperature = getattr(config, 'temperature', 1.0)
        self.logits_processor = LogitsProcessorList([TemperatureSoftmaxLogitsProcessor(self.temperature)])

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        # Clamp logits to a safe range before softmax
        logits = torch.clamp(logits, min=-1e6, max=1e6)  # Adjust range if needed

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )

    def generate(self, *args, **kwargs):
        kwargs['logits_processor'] = self.logits_processor
        return super().generate(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.transformer.enable_gradient_checkpointing()
