# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )

        self.truncation_enable = _get_env_bool("TRUNCATION_ENABLE", False)
        self.truncation_threshold_mode = os.getenv("TRUNCATION_THRESHOLD_MODE", "absolute").strip().lower()
        self.truncation_threshold = _get_env_float("TRUNCATION_THRESHOLD", 2.0)
        self.truncation_ratio = _get_env_float("TRUNCATION_RATIO", 0.1)
        self.truncation_mask_mode = os.getenv("TRUNCATION_MASK_MODE", "after_first_high").strip().lower()
        self.truncation_include_trigger = _get_env_bool("TRUNCATION_INCLUDE_TRIGGER", True)
        self.truncation_min_prefix_tokens = _get_env_int("TRUNCATION_MIN_PREFIX_TOKENS", 0)
        self.truncation_max_mask_ratio = _get_env_float("TRUNCATION_MAX_MASK_RATIO", 1.0)
        self.truncation_min_high_run = _get_env_int("TRUNCATION_MIN_HIGH_RUN", 1)
        self.truncation_random_ratio = _get_env_float("TRUNCATION_RANDOM_RATIO", 0.1)
        self.truncation_log_every_n_steps = _get_env_int("TRUNCATION_LOG_EVERY_N_STEPS", 50)
        self.truncation_log_detailed = _get_env_bool("TRUNCATION_LOG_DETAILED", False)
        self.truncation_log_detailed_every_n_steps = _get_env_int("TRUNCATION_LOG_DETAILED_EVERY_N_STEPS", 100)
        self.truncation_debug = _get_env_bool("TRUNCATION_DEBUG", False)

        self.truncation_ratio = max(0.0, min(1.0, self.truncation_ratio))
        self.truncation_random_ratio = max(0.0, min(1.0, self.truncation_random_ratio))
        self.truncation_max_mask_ratio = max(0.0, min(1.0, self.truncation_max_mask_ratio))
        self.truncation_min_prefix_tokens = max(0, self.truncation_min_prefix_tokens)
        self.truncation_min_high_run = max(1, self.truncation_min_high_run)

        if self.truncation_threshold_mode not in ("absolute", "ratio"):
            logger.warning_rank0(
                f"Unknown TRUNCATION_THRESHOLD_MODE={self.truncation_threshold_mode}, fallback to absolute."
            )
            self.truncation_threshold_mode = "absolute"

        if self.truncation_mask_mode not in ("after_first_high", "mask_high_only", "mask_random"):
            logger.warning_rank0(f"Unknown TRUNCATION_MASK_MODE={self.truncation_mask_mode}, fallback to after_first_high.")
            self.truncation_mask_mode = "after_first_high"

        if self.truncation_enable:
            incompatible_reasons = []
            if finetuning_args.use_dft_loss:
                incompatible_reasons.append("use_dft_loss")
            if finetuning_args.use_eaft_loss:
                incompatible_reasons.append("use_eaft_loss")
            if model_args is not None and getattr(model_args, "enable_liger_kernel", False):
                incompatible_reasons.append("enable_liger_kernel")
            if model_args is not None and getattr(model_args, "use_kt", False):
                incompatible_reasons.append("use_kt")

            if incompatible_reasons:
                logger.warning_rank0(
                    "Truncation disabled due to incompatible settings: {}.".format(", ".join(incompatible_reasons))
                )
                self.truncation_enable = False

            if getattr(training_args, "label_smoothing_factor", 0.0) > 0:
                logger.warning_rank0("label_smoothing_factor > 0 detected; truncation uses plain CE loss.")

            if self.truncation_enable:
                logger.info_rank0(
                    "Truncation enabled: mode={}, threshold_mode={}, threshold={}, ratio={}.".format(
                        self.truncation_mask_mode,
                        self.truncation_threshold_mode,
                        self.truncation_threshold,
                        self.truncation_ratio,
                    )
                )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        if not self.truncation_enable or not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if logits is None or labels is None:
            logger.warning_rank0_once("Truncation fallback: missing logits or labels.")
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        shift_logits = logits[..., :-1, :].contiguous() #因为后面用了 .view(...)，view 只接受连续内存。切片后的 tensor 可能是非连续的，用 contiguous() 保证内存布局连续，避免报错或隐式拷贝。
        shift_labels = labels[..., 1:].contiguous()
            # - logits 形状是 (batch, seq_len, vocab).
            # - 做 next‑token prediction，要用 t 的 logits 对齐 t+1 的 label，所以 logits 去掉最后一个 time step：:-1。
            # - 逗号是为了显式保留最后一维 vocab：logits[..., :-1, :] 得到形状 (batch, seq_len-1, vocab)。
            # - 如果写成 logits[..., :-1]，会切到 最后一维（vocab），变成 (batch, seq_len, vocab-1)，这是错的。
        vocab_size = shift_logits.size(-1)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)

        truncation_mask = self._build_truncation_mask(per_token_loss, shift_labels)
        valid_mask = shift_labels != IGNORE_INDEX
        final_mask = valid_mask & (~truncation_mask)
        final_mask_float = final_mask.to(per_token_loss.dtype)
        final_count = final_mask_float.sum()
        if final_count.item() == 0:
            loss = torch.tensor(0.0, device=per_token_loss.device, requires_grad=True)
        else:
            loss = (per_token_loss * final_mask_float).sum() / final_count

        self._log_truncation_stats(per_token_loss, valid_mask, truncation_mask)
        self._log_truncation_detailed(inputs, valid_mask, truncation_mask)
        if return_outputs:
            return loss, outputs
        return loss

    def _build_truncation_mask(self, per_token_loss: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            
            # - per_token_loss 需要梯度（对 logits 反传），但 生成 mask 的逻辑不需要梯度。
            # - 在 _build_truncation_mask(...) 里用了 with torch.no_grad()，这让阈值比较、argmax、conv1d 这些操作不进入计算图，减少显存和计算图大小。
            # - 结果就是：梯度只通过 “被保留的 token 的 loss” 反传，mask 作为常量。

            valid_mask = shift_labels != IGNORE_INDEX
            truncation_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            if self.truncation_mask_mode == "mask_random":
                return self._mask_random(valid_mask, truncation_mask)

            high_loss_positions = self._compute_high_loss_positions(per_token_loss, valid_mask)
            if self.truncation_mask_mode == "mask_high_only":
                return self._mask_high_only(high_loss_positions, valid_mask, truncation_mask)
            if self.truncation_mask_mode == "after_first_high":
                return self._mask_after_first_high(high_loss_positions, valid_mask, truncation_mask)

            return truncation_mask

    def _compute_high_loss_positions(
        self, per_token_loss: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.truncation_threshold_mode == "ratio":
            return self._high_loss_by_ratio(per_token_loss, valid_mask)
        return (per_token_loss > self.truncation_threshold) & valid_mask

    def _high_loss_by_ratio(self, per_token_loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_losses = per_token_loss[valid_mask]
        if valid_losses.numel() == 0 or self.truncation_ratio <= 0.0:
            return torch.zeros_like(valid_mask, dtype=torch.bool)
        if self.truncation_ratio >= 1.0:
            return valid_mask.clone()

        k = int(math.ceil((1.0 - self.truncation_ratio) * valid_losses.numel()))
        k = max(1, min(k, valid_losses.numel()))
        threshold = valid_losses.kthvalue(k).values
        return (per_token_loss >= threshold) & valid_mask

    def _mask_after_first_high(
        self, high_loss_positions: torch.Tensor, valid_mask: torch.Tensor, truncation_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = valid_mask.size(0)
        for i in range(batch_size):
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue

            seq_high = high_loss_positions[i, valid_indices]
            trigger_idx = self._first_high_run_index(seq_high, self.truncation_min_high_run)
            if trigger_idx is None:
                continue

            start_valid = trigger_idx + (0 if self.truncation_include_trigger else 1)
            min_prefix = min(self.truncation_min_prefix_tokens, valid_indices.numel())
            min_keep = 0
            if self.truncation_max_mask_ratio < 1.0:
                min_keep = int(math.ceil(valid_indices.numel() * (1.0 - self.truncation_max_mask_ratio)))
            start_valid = max(start_valid, min_prefix, min_keep)
            if start_valid >= valid_indices.numel():
                continue

            truncation_mask[i, valid_indices[start_valid:]] = True

        return truncation_mask

    def _mask_high_only(
        self, high_loss_positions: torch.Tensor, valid_mask: torch.Tensor, truncation_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = valid_mask.size(0)
        for i in range(batch_size):
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue

            seq_high = high_loss_positions[i, valid_indices]
            if self.truncation_min_prefix_tokens > 0:
                prefix_len = min(self.truncation_min_prefix_tokens, valid_indices.numel())
                seq_high[:prefix_len] = False
            truncation_mask[i, valid_indices[seq_high]] = True

        return truncation_mask

    def _mask_random(self, valid_mask: torch.Tensor, truncation_mask: torch.Tensor) -> torch.Tensor:
        if self.truncation_random_ratio <= 0.0:
            return truncation_mask

        batch_size = valid_mask.size(0)
        for i in range(batch_size):
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue

            if self.truncation_min_prefix_tokens > 0:
                prefix_len = min(self.truncation_min_prefix_tokens, valid_indices.numel())
                valid_indices = valid_indices[prefix_len:]
            if valid_indices.numel() == 0:
                continue

            num_to_mask = int(valid_indices.numel() * self.truncation_random_ratio)
            if num_to_mask <= 0:
                continue

            perm = torch.randperm(valid_indices.numel(), device=valid_mask.device)[:num_to_mask]
            truncation_mask[i, valid_indices[perm]] = True

        return truncation_mask

    def _first_high_run_index(self, high_mask: torch.Tensor, min_run: int) -> Optional[int]:
        if min_run <= 1:
            if high_mask.any():
                return int(torch.argmax(high_mask.float()).item())
            return None

        if high_mask.numel() < min_run or high_mask.sum().item() < min_run:
            return None

        kernel = torch.ones(min_run, device=high_mask.device, dtype=high_mask.dtype).view(1, 1, -1)
        run_sum = F.conv1d(high_mask.float().view(1, 1, -1), kernel).view(-1)
        run_mask = run_sum >= min_run
        if not run_mask.any():
            return None
        return int(torch.argmax(run_mask.float()).item())

    def _should_log_truncation(self, every_n_steps: int) -> bool:
        if every_n_steps <= 0:
            return False
        if not hasattr(self, "state"):
            return False
        return self.state.global_step % every_n_steps == 0

    def _log_truncation_stats(
        self, per_token_loss: torch.Tensor, valid_mask: torch.Tensor, truncation_mask: torch.Tensor
    ) -> None:
        if not self._should_log_truncation(self.truncation_log_every_n_steps):
            return
        if not self.is_world_process_zero():
            return

        with torch.no_grad():
            valid_counts = valid_mask.sum(dim=1)
            trunc_counts = truncation_mask.sum(dim=1)
            valid_total = valid_counts.sum().clamp_min(1)
            trunc_total = trunc_counts.sum()
            batch_ratio = (trunc_total / valid_total).item()

            seq_ratio = trunc_counts.float() / valid_counts.clamp_min(1).float()
            metrics = {
                "truncation/batch_ratio": batch_ratio,
                "truncation/mean_seq_ratio": seq_ratio.mean().item(),
                "truncation/max_seq_ratio": seq_ratio.max().item(),
                "truncation/truncated_seqs": (trunc_counts > 0).sum().item(),
            }

            raw_loss = (per_token_loss * valid_mask.float()).sum() / valid_total
            raw_ppl = torch.exp(raw_loss.clamp(max=20.0))
            metrics["truncation/raw_loss"] = raw_loss.item()
            metrics["truncation/raw_ppl"] = raw_ppl.item()

            if self.truncation_debug and self.truncation_mask_mode == "after_first_high":
                start_valid = valid_counts - trunc_counts
                prefix_violations = ((trunc_counts > 0) & (start_valid < self.truncation_min_prefix_tokens)).sum()
                max_ratio = self.truncation_max_mask_ratio + 1e-6
                max_ratio_violations = (seq_ratio > max_ratio).sum()
                metrics["truncation/prefix_violations"] = prefix_violations.item()
                metrics["truncation/max_ratio_violations"] = max_ratio_violations.item()

            self.log(metrics)

    def _log_truncation_detailed(
        self, inputs: dict[str, Union["torch.Tensor", Any]], valid_mask: torch.Tensor, truncation_mask: torch.Tensor
    ) -> None:
        if not self.truncation_log_detailed:
            return
        if self.truncation_mask_mode != "after_first_high":
            return
        if not self._should_log_truncation(self.truncation_log_detailed_every_n_steps):
            return
        if not self.is_world_process_zero():
            return
        if not hasattr(self.args, "output_dir"):
            return

        with torch.no_grad():
            valid_counts = valid_mask.sum(dim=1)
            trunc_counts = truncation_mask.sum(dim=1)
            if (trunc_counts > 0).sum().item() == 0:
                return

            log_file = os.path.join(self.args.output_dir, "truncation_detailed.jsonl")
            input_ids = inputs.get("input_ids")
            for i in range(valid_mask.size(0)):
                if trunc_counts[i].item() == 0:
                    continue

                valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(1)
                if valid_indices.numel() == 0:
                    continue

                start_valid = int(valid_counts[i].item() - trunc_counts[i].item())
                if start_valid >= valid_indices.numel():
                    continue

                entry = {
                    "step": int(getattr(self.state, "global_step", 0)),
                    "seq_id": int(i),
                    "valid_length": int(valid_counts[i].item()),
                    "truncated_tokens": int(trunc_counts[i].item()),
                    "truncate_start_idx": int(valid_indices[start_valid].item()),
                    "truncate_start_valid": int(start_valid),
                }
                if input_ids is not None:
                    entry["input_ids"] = input_ids[i].detach().cpu().tolist()

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
