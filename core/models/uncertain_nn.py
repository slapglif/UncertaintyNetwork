# core/models/uncertain_nn.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.embedding import (
    PositionalEncoding,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    StableEmbedding,
)
from core.models.layers import TransformerEncoderLayer, TimestepNorm


class UncertainTransformerConfig(PretrainedConfig):
    model_type = "uncertain_transformer"

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,  # Reduced from 768
        n_heads=8,    # Reduced from 12
        d_ff=2048,    # Reduced from 3072
        n_layers=6,   # Reduced from 12
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
        num_groups=16,  # Reduced from 32
        cema_hidden_dim=32,  # Reduced from 64
        z_dim=512,  # Reduced from 768
        v_dim=512,  # Reduced from 768
        chunk_size=64,  # Added chunk_size parameter
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


from torch.utils.checkpoint import checkpoint

class UncertainTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = StableEmbedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.dropout, config.max_position_embeddings
        )

        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                config.d_model, config.max_position_embeddings
            )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = TimestepNorm(config.num_groups, config.d_model)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embedding(input_ids)
        hidden_states = self.pos_encoding(embedding_output)

        if self.config.use_rotary_embeddings:
            seq_len = input_ids.size(1)
            cos, sin = self.rotary_emb(input_ids, seq_len=seq_len)
            hidden_states, _ = apply_rotary_pos_emb(
                hidden_states, hidden_states, cos, sin
            )

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, extended_attention_mask)
            else:
                hidden_states = layer(hidden_states, extended_attention_mask)

        return self.norm(hidden_states)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embedding.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs

        hidden_states = hidden_states + 1e-8
        lm_logits = self.lm_head(hidden_states)
        lm_logits = torch.clamp(lm_logits, min=-100, max=100)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
            attentions=None,
        )

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.transformer.disable_gradient_checkpointing()