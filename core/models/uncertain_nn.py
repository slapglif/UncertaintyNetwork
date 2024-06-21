# core/models/uncertain_nn.py

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.embedding import (
    PositionalEncoding,
    RotaryEmbedding,
)
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
            use_rotary_embeddings=True,
            use_stable_embedding=True,
            num_groups=16,
            cema_hidden_dim=32,
            z_dim=512,
            v_dim=512,
            chunk_size=64,
            use_gelu_approximation=True,
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
        self.tie_word_embeddings = tie_word_embeddings
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_stable_embedding = use_stable_embedding
        self.num_groups = num_groups
        self.cema_hidden_dim = cema_hidden_dim
        self.z_dim = z_dim
        self.v_dim = v_dim
        self.chunk_size = chunk_size
        self.use_gelu_approximation = use_gelu_approximation


class UncertainTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_position_embeddings)

        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(config.d_model, config.max_position_embeddings)

        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embedding(input_ids)
        hidden_states = self.pos_encoding(embedding_output)

        all_hidden_states = () if self.config.output_hidden_states else None
        all_attentions = () if self.config.output_attentions else None
        past_key_values = past_key_values or (None,) * len(self.layers)

        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask,
                )
            else:
                hidden_states = layer(hidden_states, extended_attention_mask, layer_past)

            if self.config.output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.output_attentions:
                all_attentions += (layer.self_attn.attn_probs,)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        self.lm_head.weight = self.transformer.embedding.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """
        Forward pass of the UncertainTransformerLMHeadModel.

        Args:
            input_ids (Optional[torch.LongTensor]): Indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[torch.FloatTensor]): Mask to avoid performing attention on padding token indices.
            labels (Optional[torch.LongTensor]): Labels for computing the masked language modeling loss.

        Returns:
            Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]: A tuple or CausalLMOutputWithCrossAttentions
                containing the loss, logits, hidden states, and attentions.
        """
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """
        Prepare inputs for generation.

        Args:
            input_ids (torch.LongTensor): Input ids for the current step.
            past (Optional[Tuple]): Past key/value states for attention.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared inputs.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "past_key_values": past,
        }

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.transformer.disable_gradient_checkpointing()
