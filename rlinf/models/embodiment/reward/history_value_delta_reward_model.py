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
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model as get_mlp_policy
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class HistoryValueDeltaRewardModel(BaseRewardModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        actor_model_cfg = cfg.get("actor_model_cfg", None)
        if actor_model_cfg is None:
            raise ValueError("history_value_delta requires actor_model_cfg.")

        self.actor_model_cfg = OmegaConf.create(actor_model_cfg)
        if self.actor_model_cfg.get("model_type") != "mlp_policy":
            raise ValueError(
                "history_value_delta currently supports actor.model.model_type == 'mlp_policy' only."
            )
        if not self.actor_model_cfg.get("add_value_head", False):
            raise ValueError(
                "history_value_delta requires actor model with add_value_head=True."
            )

        self.positive_delta_threshold = float(cfg.get("positive_delta_threshold", 0.2))
        self.negative_delta_threshold = float(cfg.get("negative_delta_threshold", -0.2))
        self.neutral_reward = float(cfg.get("neutral_reward", 0.0))

        self.value_model = get_mlp_policy(self.actor_model_cfg)
        self._load_actor_checkpoint()
        self.value_model.eval()

    def _resolve_checkpoint_path(self) -> Path:
        checkpoint_path = self.cfg.get("value_model_checkpoint_path", None)
        if checkpoint_path is None:
            raise ValueError("history_value_delta requires value_model_checkpoint_path.")

        checkpoint_path = Path(checkpoint_path)
        candidate_paths = [
            checkpoint_path,
            checkpoint_path / "model_state_dict" / "full_weights.pt",
            checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt",
        ]
        for candidate_path in candidate_paths:
            if candidate_path.is_file():
                return candidate_path
        raise FileNotFoundError(
            f"Could not find actor checkpoint weights from {checkpoint_path}."
        )

    def _load_actor_checkpoint(self) -> None:
        checkpoint_path = self._resolve_checkpoint_path()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        normalized_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ["module.", "_orig_mod."]:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            normalized_state_dict[normalized_key] = value
        self.value_model.load_state_dict(normalized_state_dict, strict=True)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryValueDeltaRewardModel is an inference-time reward model; training via forward() is not supported."
        )

    @torch.no_grad()
    def compute_reward(self, reward_input: dict[str, Any]) -> torch.Tensor:
        history_value_start_states = reward_input.get("history_value_start_states")
        history_value_end_states = reward_input.get("history_value_end_states")
        if history_value_start_states is None or history_value_end_states is None:
            raise ValueError(
                "HistoryValueDeltaRewardModel requires history_value_start_states and history_value_end_states."
            )

        device = next(self.value_model.parameters()).device
        value_model_dtype = next(self.value_model.parameters()).dtype
        history_value_start_states = torch.as_tensor(
            history_value_start_states, dtype=value_model_dtype, device=device
        )
        history_value_end_states = torch.as_tensor(
            history_value_end_states, dtype=value_model_dtype, device=device
        )

        start_values = self.value_model.value_head(history_value_start_states).to(
            torch.float32
        )
        end_values = self.value_model.value_head(history_value_end_states).to(
            torch.float32
        )

        reward = torch.full_like(
            start_values.reshape(-1), fill_value=self.neutral_reward
        )
        value_delta = (end_values - start_values).reshape(-1)
        reward[value_delta > self.positive_delta_threshold] = 1.0
        reward[value_delta < self.negative_delta_threshold] = -1.0
        return reward
