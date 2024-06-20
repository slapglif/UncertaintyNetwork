import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.embedding import PositionalEncoding, RotaryEmbedding, apply_rotary_pos_emb
from core.models.layers import TransformerEncoderLayer


class UncertainTransformerConfig(PretrainedConfig):
    model_type = "uncertain_transformer"

    def __init__(
            self,
            vocab_size=50257,
            d_model=768,
            n_heads=12,
            d_ff=3072,
            n_layers=12,
            dropout=0.1,
            max_position_embeddings=1024,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            pad_token_id=50256,  # Set this to the EOS token ID for GPT-2
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=True,
            use_rotary_embeddings=True,
            rotary_dim=32,
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
        self.rotary_dim = rotary_dim


class UncertainTransformer(PreTrainedModel):
    config_class = UncertainTransformerConfig

    def __init__(self, config: UncertainTransformerConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_position_embeddings)

        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(config.rotary_dim, config.max_position_embeddings)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embedding(input_ids)
        hidden_states = self.pos_encoding(embedding_output)

        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(input_ids, seq_len=input_ids.size(1))
            hidden_states = apply_rotary_pos_emb(hidden_states, hidden_states, cos, sin)

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        output = self.norm(hidden_states)
        return output


class UncertainTransformerLMHeadModel(PreTrainedModel):
    config_class = UncertainTransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config: UncertainTransformerConfig):
        super().__init__(config)
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
            attentions=None,
        )

    def generate(self, input_ids, max_length, **kwargs):
        return self.generate(input_ids, max_length=max_length, **kwargs)
