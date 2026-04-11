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

from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class HistoryEnvDeltaRewardModel(BaseRewardModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.positive_delta_threshold = float(cfg.get("positive_delta_threshold", 0.2))
        self.negative_delta_threshold = float(cfg.get("negative_delta_threshold", -0.2))
        self.neutral_reward = float(cfg.get("neutral_reward", 0.0))
        if self.positive_delta_threshold < self.negative_delta_threshold:
            raise ValueError(
                "positive_delta_threshold must be >= negative_delta_threshold."
            )

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryEnvDeltaRewardModel is an inference-time reward model; training via forward() is not supported."
        )

    @torch.no_grad()
    def compute_reward(self, reward_input: dict[str, Any]) -> torch.Tensor:
        history_env_reward_start = reward_input.get("history_env_reward_start")
        history_env_reward_end = reward_input.get("history_env_reward_end")
        if history_env_reward_start is None or history_env_reward_end is None:
            raise ValueError(
                "HistoryEnvDeltaRewardModel requires history_env_reward_start and history_env_reward_end."
            )

        history_env_reward_start = torch.as_tensor(
            history_env_reward_start, dtype=torch.float32, device="cpu"
        )
        history_env_reward_end = torch.as_tensor(
            history_env_reward_end, dtype=torch.float32, device="cpu"
        )

        reward = torch.full_like(
            history_env_reward_start, fill_value=self.neutral_reward
        )
        reward_delta = history_env_reward_end - history_env_reward_start
        reward[reward_delta >= self.positive_delta_threshold] = 1.0
        reward[reward_delta <= self.negative_delta_threshold] = 0.0
        return reward.flatten()
