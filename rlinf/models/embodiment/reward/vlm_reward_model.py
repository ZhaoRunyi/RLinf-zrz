#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from pathlib import Path

import torch

from omegaconf import DictConfig

from typing import Any, Optional

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)

from rlinf.config import torch_dtype_from_precision

from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.debug_video import (
    render_debug_video,
)
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class VLMRewardModel(BaseRewardModel):
    """A frozen VLM reward model that maps (images, task) -> scalar reward.

    This implementation intentionally avoids hardcoding family-specific HF class
    names. It loads by `model_path` via Auto* APIs (consistent with RLinf SFT).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError("reward.model.model_path must be set for VLMRewardModel")
        self.lora_path = self.cfg.get("lora_path")

        self.dtype = torch_dtype_from_precision(cfg.precision)

        self.setup_processor()
        self.setup_model()

        self.setup_input_builder()
        self.setup_reward_parser()

        self.gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
            "do_sample": bool(cfg.get("do_sample", True)),
            "temperature": float(cfg.get("temperature", 0.0))
        }
        
    def setup_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._setup_subprocessor( 
            subprocessor_kwargs=self.cfg.get("subprocessor_kwargs", {})
        )

    def _setup_subprocessor(
        self,
        subprocessor_kwargs: dict,
    ) -> None:
        for subprocessor_name, subprocessor_kwargs in subprocessor_kwargs.items():
            subprocessor_kwargs = dict(subprocessor_kwargs)

            subprocessoror = getattr(self._processor, subprocessor_name, None)
            if subprocessoror is None:
                return
            for key, value in dict(subprocessor_kwargs).items():
                if hasattr(subprocessoror, key):
                    setattr(subprocessoror, key, value)


    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(self.cfg.get("input_builder_name", "base_vlm_input_builder"))(**self.cfg.get("input_builder_params", {}), _processor=self._processor)

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(self.cfg.get("reward_parser_name", "base_reward_parser"))(**self.cfg.get("reward_parser_params", {}))

    def forward(self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, Any]:
        raise NotImplementedError(
            "VLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )
    
    def setup_model(self) -> None:
        _ = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        if self.lora_path:
            full_weights_path = os.path.join(self.lora_path, "actor", "model_state_dict", "full_weights.pt")

            checkpoint_state_dict = torch.load(
                full_weights_path,
                map_location="cpu",
                weights_only=True,
            )
            lora_state_dict = {
                key.removeprefix("module."): value
                for key, value in checkpoint_state_dict.items()
                if "lora_" in key
            }
            del checkpoint_state_dict

            lora_rank = next(
                int(value.shape[0])
                for key, value in lora_state_dict.items()
                if "lora_A" in key
            )
            target_modules = sorted(
                {
                    key.split(".lora_")[0].split(".")[-1]
                    for key in lora_state_dict
                    if ".lora_" in key
                }
            )

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=0.0,
                target_modules=target_modules,
                init_lora_weights="gaussian",
            )
            self._model = get_peft_model(self._model, lora_config)
            set_peft_model_state_dict(self._model, lora_state_dict)
            del lora_state_dict
        
        self._model.eval()

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        batched_inputs = self.input_builder.build_inputs(observations, self._model.device)
        prompt_length = batched_inputs["input_ids"].shape[-1]
        output_ids = self._model.generate(**batched_inputs, **self.gen_kwargs)
        del batched_inputs
        outputs = self._processor.batch_decode(
            output_ids[..., prompt_length:], skip_special_tokens=True
        )
        del output_ids
        return self.reward_parser.parse_rewards(outputs)

class HistoryVLMRewardModel(VLMRewardModel):
    def __init__(self, cfg: DictConfig):
        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(
            cfg.get("infer_micro_batch_size", 0)
        )
        self.debug_video_count = 0

        super().__init__(cfg)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "history_vlm_input_builder")
        )(**self.cfg.get("input_builder_params", {}), _processor=self._processor, history_buffer_names=self.history_buffer_names)
        assert isinstance(
            self.input_builder, HistoryVLMInputBuilder
        ), "HistoryVLMRewardModel only supports HistoryVLMInputBuilder"

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryVLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )

    def slice_history_input(
        self,
        history_input: dict[str, dict[str, list[list[Any]]]],
        start: int,
        end: int,
    ) -> dict[str, dict[str, list[list[Any]]]]:
        return {
            buffer_name: {
                history_key: env_sequences[start:end]
                for history_key, env_sequences in history_buffer.items()
            }
            for buffer_name, history_buffer in history_input.items()
        }

    def slice_observations(
        self,
        observations: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        sliced_observations = {}
        for observation_key, observation_values in observations.items():
            sliced_observations.update(
                {observation_key: observation_values[start:end]}
            )
        return sliced_observations

    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        history_input: dict[str, dict[str, list[list[Any]]]] = reward_input.pop("history_input")
        input_batch_size = len(next(iter(next(iter(history_input.values())).values())))
        observations = reward_input

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size

        reward_chunks: list[torch.Tensor] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            reward_chunk = torch.zeros((end - start,), dtype=torch.float32)

            batched_inputs, valid_input_ids = self.input_builder.build_inputs(
                micro_observations,
                self._model.device,
                micro_history_input,
            )
            if len(valid_input_ids) == 0:
                reward_chunks.append(reward_chunk)
                continue

            prompt_length = batched_inputs["input_ids"].shape[-1]
            output_ids = self._model.generate(**batched_inputs, **self.gen_kwargs)
            del batched_inputs

            decoded_outputs = self._processor.batch_decode(
                output_ids[..., prompt_length:], skip_special_tokens=True
            )
            del output_ids

            parsed_rewards = self.reward_parser.parse_rewards(decoded_outputs).to(dtype=torch.float32)
            debug_video_output_dir = self.cfg.get("debug_video_output_dir", None)
            if debug_video_output_dir:
                reward_worker_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
                history_buffer_name = self.cfg.get("debug_video_history_buffer_name", self.history_buffer_names[0])
                history_buffer = micro_history_input.get(history_buffer_name, {})
                history_key = self.cfg.get("debug_video_history_key", next(iter(history_buffer), None))
                history_sequences = history_buffer.get(history_key, []) if history_key is not None else []
                task_descriptions = micro_observations.get("task_descriptions", [""] * (end - start))
                for decoded_output, parsed_reward, valid_input_id in zip(decoded_outputs, parsed_rewards, valid_input_ids):
                    if valid_input_id >= len(history_sequences) or not history_sequences[valid_input_id]:
                        continue
                    local_env_id = start + valid_input_id
                    output_path = Path(debug_video_output_dir) / (
                        f"rank_{reward_worker_rank}_env_{local_env_id:04d}_count_{self.debug_video_count:06d}.mp4"
                    )
                    render_debug_video(
                        history_frames=history_sequences[valid_input_id],
                        footer_lines=[
                            f"pred_reward: {float(parsed_reward.item()):.4f}",
                            f"task: {task_descriptions[valid_input_id]}",
                            f"raw_output: {decoded_output.strip() or '<empty>'}",
                        ],
                        output_path=output_path,
                        fps=int(self.cfg.get("debug_video_fps", 4)),
                    )
                    self.debug_video_count += 1

            reward_chunk[valid_input_ids] = parsed_rewards
            reward_chunks.append(reward_chunk)
            del decoded_outputs

        return torch.cat(reward_chunks, dim=0)
        
