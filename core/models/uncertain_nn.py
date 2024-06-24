# core/models/uncertain_nn.py

from typing import Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from core.models.embedding import SentenceGP, SentenceEncoder
from core.models.layers import TransformerEncoderLayer, MambaLayer


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
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_mamba=True,
            d_state=16,
            d_conv=4,
            expand_factor=2.0,
            dt_rank=None,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            use_flash_attention=False,
            n_inducing=10,
            use_gelu_approximation=False,
            sliding_window_size=512,
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
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.use_flash_attention = use_flash_attention
        self.n_inducing = n_inducing
        self.use_gelu_approximation = use_gelu_approximation
        self.sliding_window_size = sliding_window_size

        # Initialize _no_split_modules as an empty list
        self._no_split_modules = []


class UncertainNN(nn.Module):
    def __init__(self, config: UncertainTransformerConfig | PretrainedConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

        # Interleave Mamba and Transformer layers (Samba-like)
        self.layers = nn.ModuleList(
            [
                MambaLayer(config) if i % 2 == 0 else TransformerEncoderLayer(config)
                for i in range(config.n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

        # Components for uncertainty modeling
        self.sentence_encoder = SentenceEncoder(
            config.d_model, config.d_model * 2, config.d_model
        )
        self.sentence_gp = SentenceGP(
            config.d_model, config.d_model, config.n_inducing, config.d_model
        )

        self.sliding_window_size = config.sliding_window_size

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            token_type_ids: Any = None,
            head_mask: Any = None,
            encoder_hidden_states: Any = None,
            encoder_attention_mask: Any = None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        padding_needed = 0
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        seq_length = (
            input_shape[-1] if input_ids is not None else inputs_embeds.shape[1]
        )

        if position_ids is None:
            device = (
                input_ids.device
                if input_ids is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, (layer, past) in enumerate(
                zip(self.layers, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                if isinstance(layer, MambaLayer):
                    layer_outputs = layer(hidden_states)
                elif isinstance(layer, TransformerEncoderLayer):
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        past_key_value=past,
                    )
                else:
                    raise ValueError(
                        f"Unexpected layer type: {type(layer)}"
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[1]
                    if len(layer_outputs) > 1
                    else None,
                )

            if output_attentions and isinstance(
                    layer, TransformerEncoderLayer
            ):
                all_attentions += (layer_outputs[-1],)

        hidden_states = self.final_layer_norm(hidden_states)

        # Apply Sentence Encoder and Gaussian Process (for uncertainty)
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len >= self.config.max_position_embeddings:
            num_sentences = (
                    seq_len // self.config.max_position_embeddings
            )
            remainder = (
                    seq_len % self.config.max_position_embeddings
            )

            if remainder > 0:
                padding_needed = (
                        self.config.max_position_embeddings - remainder
                )
                hidden_states = F.pad(
                    hidden_states, (0, 0, 0, padding_needed)
                )
                seq_len += padding_needed

            hidden_states = hidden_states.view(
                batch_size,
                num_sentences,
                self.config.max_position_embeddings,
                -1,
            )
            sentence_embeddings = self.sentence_encoder(
                hidden_states
            )
            sentence_mean, sentence_var = self.sentence_gp(
                sentence_embeddings, num_sentences
            )

            if self.training:
                hidden_states = sentence_mean + torch.randn_like(
                    sentence_mean
                ) * torch.sqrt(sentence_var)
            else:
                hidden_states = sentence_mean

            hidden_states = hidden_states.view(
                batch_size, seq_len, -1
            )

            if remainder > 0:
                hidden_states = hidden_states[
                                :, : seq_len - padding_needed, :
                                ]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def _apply_sentence_encoding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        num_sentences = seq_len // self.config.max_position_embeddings
        remainder = seq_len % self.config.max_position_embeddings
        padding_needed = 0
        if remainder > 0:
            padding_needed = self.config.max_position_embeddings - remainder
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_needed))
            seq_len += padding_needed

        hidden_states = hidden_states.view(
            batch_size, num_sentences, self.config.max_position_embeddings, -1
        )
        sentence_embeddings = self.sentence_encoder(hidden_states)
        sentence_mean, sentence_var = self.sentence_gp(sentence_embeddings, num_sentences)

        if self.training:
            hidden_states = sentence_mean + torch.randn_like(sentence_mean) * torch.sqrt(sentence_var)
        else:
            hidden_states = sentence_mean

        hidden_states = hidden_states.view(batch_size, seq_len, -1)

        if remainder > 0:
            hidden_states = hidden_states[:, :seq_len - padding_needed, :]

        return hidden_states

    @classmethod
    def _gradient_checkpointing_func(cls, func, *args, **kwargs):
        return checkpoint(func, *args, **kwargs)


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.transformer = UncertainNN(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Set the model's parameters
        self.main_input_name = "input_ids"
        self.config.is_decoder = True
        self.config.tie_word_embeddings = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # update attention mask
        if past_key_values is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding = value

    def tie_weights(self):
        self.lm_head.weight = self.transformer.embedding.weight
