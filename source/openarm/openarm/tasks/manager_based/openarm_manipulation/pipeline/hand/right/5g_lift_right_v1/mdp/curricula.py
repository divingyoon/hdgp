# Copyright 2025 Enactic, Inc.
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

"""Custom curriculum functions for 5g_lift_right tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class linear_reward_weight(ManagerTermBase):
    """Curriculum that linearly interpolates a reward weight over a step range."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        t = env.common_step_counter

        if t <= start_step:
            w = start_weight
        elif t >= end_step:
            w = end_weight
        else:
            alpha = (t - start_step) / (end_step - start_step)
            w = start_weight + alpha * (end_weight - start_weight)

        self._term_cfg.weight = w
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return w
