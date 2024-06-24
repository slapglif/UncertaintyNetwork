# core/models/uncertain_nn.py

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from core.models.embedding import RotaryPositionEncoding, SentenceEncoder, SentenceGP, apply_rotary_pos_emb
from core.models.layers import TransformerEncoderLayer, CEMA, KANFeedForward
from core.models.mamba import Mamba, MambaConfig
from core.utils.uncertainty import UncertaintyModule


class UncertainTransformerConfig(PretrainedConfig):
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
            expand_factor=2,
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


class UncertainNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_pos_emb = RotaryPositionEncoding(config.d_model, config.n_heads, config.max_position_embeddings)
        self.sentence_encoder = SentenceEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.d_model,
            output_dim=config.d_model,
            kan_config={
                'layers_hidden': [1024, 2048],
                'grid_min': -1.2,
                'grid_max': 0.2,
                'num_grids': 8,
                'exponent': 2,
                'inv_denominator': 0.5,
                'train_grid': False,
                'train_inv_denominator': False,
                'spline_weight_init_scale': 1.0,
                'uncertainty_output': True,
            }
        )
        self.sentence_gp = SentenceGP(config.d_model, config.d_model, config.n_inducing, config.d_model)
        self.gp_projection = nn.Linear(config.n_inducing, config.d_model)
        self.cema = CEMA(config.d_model)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Mamba(MambaConfig(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand_factor=config.expand_factor,
                    d_conv=config.d_conv,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor
                )),
                TransformerEncoderLayer(config),
                KANFeedForward(config),
                nn.Linear(config.d_model, config.d_model)
            ])
            for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.uncertainty_module = UncertaintyModule(
            input_dim=config.d_model,
            output_dim=config.vocab_size,
            n_gp_layers=2,
            n_inducing=config.n_inducing,
            dropout_rate=config.dropout
        )

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
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

            sentence_emb, sentence_uncertainty = self.sentence_encoder(input_ids)
            gp_mean, gp_var, _ = self.sentence_gp(sentence_emb)
            gp_mean = self.gp_projection(gp_mean)

            inputs_embeds = inputs_embeds + sentence_emb + gp_mean

            cos, sin = self.rotary_pos_emb(inputs_embeds, seq_len=seq_length)
            inputs_embeds = apply_rotary_pos_emb(inputs_embeds, cos, sin)

        inputs_embeds = self.cema(inputs_embeds)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        mamba_states = [None] * len(self.layers) if past_key_values is None else [
            cache[0] for cache in past_key_values
        ]

        for i, (mamba, transformer, kan_ff, projection) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            mamba_state = mamba_states[i]

            hidden_states, mamba_cache = mamba(hidden_states, mamba_state, use_cache=use_cache)
            hidden_states = projection(hidden_states)

            layer_outputs = transformer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            hidden_states = kan_ff(hidden_states)

            if use_cache:
                next_decoder_cache += ((mamba_cache, layer_outputs[1]),)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)

        # Apply uncertainty module
        mean_output, uncertainty = self.uncertainty_module(hidden_states)

        # Combine uncertainties
        combined_uncertainty = uncertainty + sentence_uncertainty + gp_var

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in
                         [mean_output, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
                         if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=mean_output,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), combined_uncertainty


class UncertainTransformerLMHeadModel(PreTrainedModel):
    def __init__(self, config: 'UncertainTransformerConfig'):
        super().__init__(config)
        self.transformer = UncertainNN(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, value):
        self.transformer.embedding = value

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

        transformer_outputs, uncertainty = self.transformer(
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

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        ), uncertainty

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            mamba_state, transformer_past = layer_past
            reordered_mamba_state = tuple(past_state.index_select(0, beam_idx) for past_state in mamba_state)
            reordered_transformer_past = tuple(past_state.index_select(0, beam_idx) for past_state in transformer_past)
            reordered_past += ((reordered_mamba_state, reordered_transformer_past),)
        return reordered_past
