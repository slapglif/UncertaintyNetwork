# core/models/uncertain_nn.py
import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.layers import MultiHeadAttention, PositionwiseFeedForward


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
            **kwargs
    ):
        super().__init__(pad_token_id=0, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings


class UncertainTransformerEncoderLayer(nn.Module):
    def __init__(self, config: UncertainTransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class UncertainTransformer(PreTrainedModel):
    config_class = UncertainTransformerConfig

    def __init__(self, config: UncertainTransformerConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_position_embeddings)
        self.encoder_layers = nn.ModuleList(
            [UncertainTransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        position_embeds = self.pos_encoding(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)

        return hidden_states


class UncertainTransformerLMHeadModel(PreTrainedModel):
    config_class = UncertainTransformerConfig

    def __init__(self, config: UncertainTransformerConfig):
        super().__init__(config)
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # Only last token for inputs_ids if past is defined in kwargs
        if past is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
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
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
