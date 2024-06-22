from typing import Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.checkpoint import checkpoint
from torchmetrics.text import Perplexity
from transformers import PreTrainedModel, PretrainedConfig
from transformers import get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from core.models.embedding import PositionalEncoding, StableEmbedding
from core.models.layers import TransformerEncoderLayer


class UncertainTransformerConfig(PretrainedConfig):
    model_type = "uncertain_transformer"

    def __init__(
            self,
            vocab_size: int = 50257,
            d_model: int = 384,
            n_heads: int = 6,
            d_ff: int = 1536,
            n_layers: int = 4,
            dropout: float = 0.1,
            max_position_embeddings: int = 512,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = True,
            pad_token_id: int = 50256,
            bos_token_id: int = 50256,
            eos_token_id: int = 50256,
            tie_word_embeddings: bool = True,
            use_stable_embedding: bool = True,
            cema_hidden_dim: int = 64,
            chunk_size: int = 64,
            use_gelu_approximation: bool = True,
            max_grad_norm: float = 1.0,
            ntk_factor: float = 1.0,
            uncertainty_factor: float = 0.1,
            use_flash_attention: bool = True,
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


class UncertainTransformer(nn.Module):
    def __init__(self, config: UncertainTransformerConfig):
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

        if self.gradient_checkpointing and self.training:
            hidden_states = checkpoint(self._forward_transformer, hidden_states, attention_mask, position_ids,
                                       past_key_values, use_cache, output_attentions, output_hidden_states)
        else:
            hidden_states = self._forward_transformer(hidden_states, attention_mask, position_ids, past_key_values,
                                                      use_cache, output_attentions, output_hidden_states)

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

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


class UncertainTransformerLMHeadModel(PreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config: UncertainTransformerConfig):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.transformer = UncertainTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.perplexity = Perplexity(ignore_index=-100)
        self.tokenizer = None
        self.gradient_checkpointing = False
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
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            transformer_outputs = checkpoint(self.transformer, input_ids, attention_mask, position_ids, past_key_values,
                                             use_cache, output_attentions, output_hidden_states, return_dict)
        else:
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

        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        perplexity = self.perplexity(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx == 0:
            input_ids = batch[0][:1]
            generated = self.generate(input_ids, max_length=50)
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            if isinstance(self.logger, TensorBoardLogger) and hasattr(self.logger.experiment, 'add_text'):
                self.logger.experiment.add_text("generated_text", generated_text, self.current_epoch)

        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if all(nd not in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.transformer.enable_gradient_checkpointing()
