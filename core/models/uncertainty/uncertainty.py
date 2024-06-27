# .\core\models\uncertainty\uncertain_nn.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from core.models.embedding import RotaryPositionEncoding, apply_rotary_pos_emb
from core.models.layers import TransformerEncoderLayer, CEMA, KANFeedForward
from core.models.statespace import Mamba, MambaConfig
from core.models.uncertainty.layers import UncertaintyModule, UncertaintyAwareLoss
from core.models.uncertainty.uncertainty_utils import uncertainty_decomposition
from core.utils.utils import TimestepNorm


class UncertainTransformerConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size: int = 50257,
            d_model: int = 768,  # Adjusted to match typical transformer models
            n_heads: int = 12,
            d_ff: int = 3072,
            n_layers: int = 12,
            dropout: float = 0.1,
            max_position_embeddings: int = 1024,
            layer_norm_epsilon: float = 1e-5,
            initializer_range: float = 0.02,
            use_cache: bool = False,
            pad_token_id: int = 50256,
            bos_token_id: int = 50256,
            eos_token_id: int = 50256,
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
            use_gradient_checkpointing: bool = False,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        self.d_inner = int(self.expand_factor * self.d_model)  # Add this line
        self.dt_rank = dt_rank if dt_rank is not None else self.d_inner
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
        self.use_gradient_checkpointing = use_gradient_checkpointing


class UncertainNetwork(nn.Module):
    def __init__(self, config: UncertainTransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model).to(config.device)
        self.rotary_pos_emb = RotaryPositionEncoding(
            config.d_model, config.n_heads, config.max_position_embeddings
        ).to(config.device)
        self.cema = CEMA(config.d_model).to(config.device)
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
        ).to(config.device)
        self.final_layer_norm = nn.LayerNorm(config.d_model).to(config.device)
        self.dropout = nn.Dropout(config.dropout).to(config.device)

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], torch.Tensor]:
        print(f"UncertainNN input shape: {inputs_embeds.shape}")
        print(f"UncertainNN input contains NaN: {torch.isnan(inputs_embeds).any()}")

        batch_size, seq_len, _ = inputs_embeds.shape
        cos, sin = self.rotary_pos_emb(inputs_embeds)
        inputs_embeds = apply_rotary_pos_emb(inputs_embeds, cos, sin)

        inputs_embeds = self.cema(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds)
        inputs_embeds = TimestepNorm(self.config.d_model).to(self.config.device)(inputs_embeds)

        hidden_states = inputs_embeds
        cache = () if use_cache else None

        for mamba, transformer, kan_ff, projection in self.layers:
            mamba_outputs = mamba(hidden_states, use_cache=use_cache)
            hidden_states = mamba_outputs[0]
            if use_cache:
                cache += (mamba_outputs[1],)

            hidden_states = projection(hidden_states)
            transformer_outputs = transformer(
                hidden_states, attention_mask=attention_mask
            )
            hidden_states = transformer_outputs[0]
            if use_cache:
                cache += transformer_outputs[1:]

            hidden_states = kan_ff(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        print(f"UncertainNN output shape: {hidden_states.shape}")
        print(f"UncertainNN output contains NaN: {torch.isnan(hidden_states).any()}")

        return hidden_states, cache, torch.zeros_like(hidden_states)  # placeholder for uncertainty

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
        self.transformer = UncertainNetwork(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.uncertainty_module = UncertaintyModule(
            input_dim=config.d_model,
            output_dim=config.vocab_size,
            n_gp_layers=1,
            n_inducing=5,
            dropout_rate=0.1,
            mc_samples=3,
        )
        self.uncertainty_loss = UncertaintyAwareLoss(uncertainty_weight=config.uncertainty_weight)
        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize weights
        self.apply(self._init_weights)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs device: {input_ids.device}")
        print(f"Input IDs contains NaN: {torch.isnan(input_ids).any()}")

        inputs_embeds = self.transformer.embedding(input_ids)
        print(f"Embedded shape: {inputs_embeds.shape}")
        print(f"Embedded contains NaN: {torch.isnan(inputs_embeds).any()}")

        hidden_states, _, _ = self.transformer(inputs_embeds, attention_mask=attention_mask)
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Hidden states contains NaN: {torch.isnan(hidden_states).any()}")

        logits = self.lm_head(hidden_states)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits contains NaN: {torch.isnan(logits).any()}")

        _, uncertainties = self.uncertainty_module(hidden_states)
        print(f"Uncertainties shape: {uncertainties.shape}")
        print(f"Uncertainties contains NaN: {torch.isnan(uncertainties).any()}")

        outputs = {"logits": logits, "uncertainties": uncertainties}

        if labels is not None:
            print(f"Labels shape: {labels.shape}")
            print(f"Labels device: {labels.device}")

            loss = self.uncertainty_loss(logits, labels, uncertainties)
            outputs["loss"] = loss

        return outputs

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _compute_embeddings(self, input_ids: torch.LongTensor) -> Tensor:
        """
        Computes sentence embeddings from input_ids using the SentenceTransformer model.

        Args:
            input_ids (torch.LongTensor): The input token IDs.

        Returns:
            torch.FloatTensor: The computed sentence embeddings.
        """
        # Decode token IDs to sentences
        sentences = self.tokenizer.batch_decode(input_ids.to(self.device), skip_special_tokens=True)

        # Embed sentences using SentenceTransformer
        embeddings = self.sentence_transformer.encode(sentences, convert_to_tensor=True).to(self.device)
        embeddings = embeddings.to(self.device)  # Ensure embeddings are on the correct device

        return embeddings

    def generate_with_uncertainty(
            self,
            input_ids: torch.LongTensor,
            max_length: int,
            num_return_sequences: int = 1,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.95,
            do_sample: bool = True,
            **_
    ) -> tuple[Tensor, Tensor]:
        """
        Generate sequences with associated uncertainties.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            max_length (int): Maximum length of generated sequences.
            num_return_sequences (int): Number of sequences to generate for each input.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p sampling parameter.
            do_sample (bool): Whether to use sampling or greedy decoding.
            **_: Additional arguments for generation.

        Returns:
            Tuple[torch.LongTensor, torch.FloatTensor]: Generated sequences and their uncertainties.
        """
        _batch_size = input_ids.shape[0]
        _device = input_ids.device

        generated_sequences = []
        sequence_uncertainties = []

        for _ in range(num_return_sequences):
            curr_input_ids = input_ids.clone()
            curr_uncertainties = []

            for _ in range(max_length - curr_input_ids.shape[1]):
                outputs = self(curr_input_ids)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token_uncertainty = outputs["uncertainties"][:, -1, :]

                # Apply temperature and uncertainty-aware sampling
                next_token_scores = next_token_logits / temperature - self.config.uncertainty_scale * next_token_uncertainty

                # Apply top-k and top-p filtering
                if do_sample:
                    next_token_scores = self._top_k_top_p_filtering(
                        next_token_scores, top_k=top_k, top_p=top_p
                    )
                    probs = torch.softmax(next_token_scores, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_scores, dim=-1)

                curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(-1)], dim=-1)
                curr_uncertainties.append(next_token_uncertainty)

            generated_sequences.append(curr_input_ids)
            sequence_uncertainties.append(torch.stack(curr_uncertainties, dim=1))

        return (
            torch.stack(generated_sequences),
            torch.stack(sequence_uncertainties)
        )

    @staticmethod
    def _top_k_top_p_filtering(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

        Args:
            logits (torch.Tensor): Logits distribution shape (batch size, vocabulary size).
            top_k (int): Keep only top k tokens with the highest probability (top-k filtering).
            top_p (float): Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            filter_value (float): Value to assign to filtered tokens.

        Returns:
            torch.Tensor: Filtered logits.
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits

    def calibrate(self, val_dataloader: DataLoader, method: str = "temperature_scaling") -> float:
        """
        Calibrate the model's uncertainty estimates.

        Args:
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            method (str): Calibration method to use. Currently only supports "temperature_scaling".

        Returns:
            float: The optimal temperature found for scaling.

        Raises:
            ValueError: If an unsupported calibration method is specified.
        """
        if method != "temperature_scaling":
            raise ValueError(f"Unsupported calibration method: {method}")

        self.eval()
        nll_criterion = nn.CrossEntropyLoss()

        def temperature_scale(t: float) -> float:
            with torch.no_grad():
                nll = 0.0
                for batch in val_dataloader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    outputs = self(input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"] / t
                    nll += nll_criterion(logits.view(-1, logits.size(-1)), labels.view(-1)).item()
            return nll

        optimal_t = minimize_scalar(temperature_scale, bounds=(0.1, 10.0), method='bounded')
        self.temperature.data = torch.tensor([optimal_t.x], device=self.device)

        return optimal_t.x

    def compute_uncertainty_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute various uncertainty metrics on a given dataset.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Dict[str, float]: Dictionary containing various uncertainty metrics.
        """
        self.eval()
        total_ece = 0.0
        total_brier = 0.0
        total_nll = 0.0
        total_samples = 0
        all_uncertainties = []
        all_accuracies = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                uncertainties = outputs["uncertainties"]

                probs = torch.nn.functional.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                accuracies = (predictions == labels).float()

                # Expected Calibration Error
                confidences, _ = torch.max(probs, dim=-1)
                ece = self._expected_calibration_error(confidences.cpu().numpy(), accuracies.cpu().numpy())

                # Brier Score
                brier = torch.mean(
                    (probs - torch.nn.functional.one_hot(labels, num_classes=self.config.vocab_size)) ** 2
                )

                # Negative Log-Likelihood
                nll = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

                batch_size = input_ids.size(0)
                total_ece += ece * batch_size
                total_brier += brier.item() * batch_size
                total_nll += nll.item() * batch_size
                total_samples += batch_size

                all_uncertainties.extend(uncertainties.mean(dim=1).cpu().numpy())
                all_accuracies.extend(accuracies.mean(dim=1).cpu().numpy())

        avg_ece = total_ece / total_samples
        avg_brier = total_brier / total_samples
        avg_nll = total_nll / total_samples

        # Compute additional metrics
        auroc = self._compute_auroc(all_uncertainties, all_accuracies)
        aupr = self._compute_aupr(all_uncertainties, all_accuracies)

        return {
            "ece": avg_ece,
            "brier_score": avg_brier,
            "nll": avg_nll,
            "auroc": auroc,
            "aupr": aupr,
        }

    @staticmethod
    def _expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute the Expected Calibration Error.

        Args:
            confidences (np.ndarray): Predicted confidences.
            accuracies (np.ndarray): True accuracies.
            n_bins (int): Number of bins for calibration.

        Returns:
            float: The Expected Calibration Error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece)

    @staticmethod
    def _compute_auroc(uncertainties: List[float], accuracies: List[float]) -> float:
        """
        Compute Area Under the Receiver Operating Characteristic curve.

        Args:
            uncertainties (List[float]): List of uncertainty values.
            accuracies (List[float]): List of accuracy values.

        Returns:
            float: The AUROC score.
        """
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(1 - np.array(accuracies), uncertainties))

    @staticmethod
    def _compute_aupr(uncertainties: List[float], accuracies: List[float]) -> float:
        """
        Compute Area Under the Precision-Recall curve.

        Args:
            uncertainties (List[float]): List of uncertainty values.
            accuracies (List[float]): List of accuracy values.

        Returns:
            float: The AUPR score.
        """
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(1 - np.array(accuracies), uncertainties))

    def uncertainty_decomposition(self, uncertainties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose total uncertainty into aleatoric and epistemic components.

        Args:
            uncertainties (torch.Tensor): Total uncertainties.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Aleatoric and epistemic uncertainties.
        """
        return uncertainty_decomposition(uncertainties, self.uncertainty_module.aleatoric_uncertainty(uncertainties))

    def active_learning_acquisition(self, pool_dataloader: DataLoader, n_samples: int) -> List[int]:
        """
        Perform active learning acquisition using uncertainty estimates.

        Args:
            pool_dataloader (DataLoader): DataLoader for the pool of unlabeled data.
            n_samples (int): Number of samples to acquire.

        Returns:
            List[int]: Indices of selected samples for labeling.
        """
        self.eval()
        uncertainties = []
        indices = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(pool_dataloader), desc="Pooling Dataloader...", total=len(pool_dataloader)):
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                outputs = self(input_ids, attention_mask=attention_mask)
                batch_uncertainties = outputs["uncertainties"].mean(dim=1)  # Average uncertainty across sequence

                uncertainties.extend(batch_uncertainties.cpu().numpy())
                indices.extend(range(i * pool_dataloader.batch_size, (i + 1) * pool_dataloader.batch_size))

        sorted_indices = [index for _, index in sorted(zip(uncertainties, indices), reverse=True)]
        return sorted_indices[:n_samples]
