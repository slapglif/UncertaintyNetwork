# core/models/uncertain_nn.py
from typing import Any, Dict
from typing import List
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import LogitsProcessor
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput

from core.models.layers import (
    TransformerEncoderLayer,
    SentenceEncoder,
    SentenceGP,
)


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
            use_gelu_approximation=False,  # Add this line
            **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
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
        self.use_gelu_approximation = use_gelu_approximation  # Add this line

    @property
    def hidden_size(self):
        return self.d_model


class UncertainNN(nn.Module):
    def __init__(self, config: UncertainTransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.n_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

        # SentenceEncoder and SentenceGP components
        self.sentence_encoder = SentenceEncoder(
            config.d_model, config.d_model * 2, config.d_model, num_grids=8
        )
        self.sentence_gp = SentenceGP(
            config.d_model, config.d_model, config.n_inducing, config.d_model
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass of the UncertainNN model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (Optional[torch.LongTensor]): Token type IDs. Not used.
            position_ids (Optional[torch.LongTensor]): Position IDs of shape (batch_size, sequence_length).
            past_key_values (Optional[Tuple[torch.FloatTensor]]): Past key values for efficient decoding.
            inputs_embeds (Optional[torch.FloatTensor]): Pre-computed input embeddings.
            use_cache (Optional[bool]): Whether to use the past key/values cache.
            output_attentions (Optional[bool]): Whether to output attention weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: Model outputs including
            last hidden state, past key values (if applicable), all hidden states (if requested),
            and attention weights (if requested).
        """
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
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(
                0, input_shape[-1], dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1:],)  # Collect all cache outputs

            if output_attentions and len(layer_outputs) > 1:
                all_attentions += (
                    layer_outputs[-1],
                )  # Get the last element as attention weights

        hidden_states = self.final_layer_norm(hidden_states)

        # Apply SentenceEncoder
        batch_size = hidden_states.shape[0]  # Get batch size directly
        seq_len = hidden_states.shape[1]  # Get sequence length directly

        # Calculate the number of sentences based on the sequence length and max_position_embeddings
        num_sentences = seq_len // self.config.max_position_embeddings
        remainder = seq_len % self.config.max_position_embeddings

        if remainder > 0:
            # Pad the sequence length to be a multiple of max_position_embeddings
            padding_needed = self.config.max_position_embeddings - remainder
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding_needed))
            seq_len += padding_needed

        # If num_sentences is 0, it means the sequence is shorter than max_position_embeddings.
        # In this case, we treat the entire sequence as one sentence.
        if num_sentences == 0:
            num_sentences = 1

        hidden_states = hidden_states.view(
            batch_size, num_sentences, self.config.max_position_embeddings, -1
        )
        sentence_embeddings = self.sentence_encoder(hidden_states)

        # Apply SentenceGP
        sentence_mean, sentence_var = self.sentence_gp(
            sentence_embeddings, num_sentences
        )

        if self.training:
            # During training, sample from the Gaussian distribution defined by sentence_mean and sentence_var
            hidden_states = sentence_mean + torch.randn_like(
                sentence_mean
            ) * torch.sqrt(sentence_var)
        else:
            # During evaluation, use the mean as the output
            hidden_states = sentence_mean

        # Flatten hidden_states back to the original shape for compatibility with subsequent layers
        hidden_states = hidden_states.view(batch_size, seq_len, -1)

        # Remove padding if added
        if remainder > 0:
            hidden_states = hidden_states[:, :seq_len - padding_needed, :]

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

    def enable_gradient_checkpointing(self):
        """
        Enables gradient checkpointing for memory efficiency during training.
        """
        self.gradient_checkpointing = True


class TemperatureSoftmaxLogitsProcessor(LogitsProcessor):
    """
    Processes the logits by applying a softmax with temperature scaling.

    Args:
        temperature (float): The temperature for scaling the logits before softmax.
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
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


class UncertainTransformerLMHeadModel(PreTrainedModel, GenerationMixin):
    """
    Language Model with a Hybrid Transformer and Sentence-Level Uncertainty.

    This model combines the UncertainNN with a language modeling head for text generation,
    incorporating uncertainty estimation at the sentence level. It uses Hugging Face's
    GenerationMixin for text generation capabilities.

    Attributes:
        config (UncertainTransformerConfig): Configuration object for the model.
        transformer (UncertainNN): The main transformer model with uncertainty estimation.
        lm_head (nn.Linear): Language modeling head for vocabulary projection.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.transformer = UncertainNN(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tie_weights()

        # This is important for generation
        self.main_input_name = "input_ids"

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.transformer.embedding.weight

    def get_output_embeddings(self) -> nn.Linear:
        """Get the output embeddings layer."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set the output embeddings layer."""
        self.lm_head = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings layer."""
        return self.transformer.embedding

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward pass of the UncertainTransformerLMHeadModel.

        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            token_type_ids (Optional[torch.LongTensor]): Token type IDs.
            position_ids (Optional[torch.LongTensor]): Position IDs.
            past_key_values (Optional[List[torch.FloatTensor]]): Past key values for efficient decoding.
            inputs_embeds (Optional[torch.FloatTensor]): Pre-computed input embeddings.
            labels (Optional[torch.LongTensor]): Labels for language modeling.
            use_cache (Optional[bool]): Whether to use the past key/values cache.
            output_attentions (Optional[bool]): Whether to output attention weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]: Model outputs including loss, logits, and other optional elements.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

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

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        """
        Prepare inputs for generation. This method is used by Hugging Face's generation utilities.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            past (Optional[List[torch.FloatTensor]]): Past key values for efficient decoding.
            attention_mask (Optional[torch.LongTensor]): Attention mask.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary of prepared inputs for generation.
        """
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        Reorder the cache for beam search. This method is used by Hugging Face's generation utilities.

        Args:
            past (List[torch.Tensor]): Past key values.
            beam_idx (torch.LongTensor): Beam indices.

        Returns:
            List[torch.Tensor]: Reordered past key values.
        """
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
            num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        if (
                model_kwargs.get("use_cache", True)
                and "cache_position" in model_kwargs
                and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = (
                    model_kwargs["cache_position"][-1:] + num_new_tokens
            )

        return model_kwargs

    def _extract_past_from_model_output(
            self, outputs: ModelOutput, standardize_cache_format: bool = False
    ):
        past = None
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems
        elif "past_buckets_states" in outputs:
            past = outputs.past_buckets_states

        # Standardize the format of the cache
        if standardize_cache_format and past is not None:
            if isinstance(past[0], tuple):
                past = tuple(tuple(torch.stack(p) for p in layer) for layer in past)
            else:
                past = tuple(p.transpose(1, 2) for p in past)

        return past
