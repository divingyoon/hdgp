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

"""유틸리티: open-tesol_r_reach_v5

순수 접근(Pure Reach) 태스크용 경량 텐서 및 수학 유틸리티 함수 모음.
"""

from __future__ import annotations

import torch


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 정규화 액션을 [lower, upper] 물리 범위로 선형 스케일링."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[lower, upper] 물리 값을 [-1, 1] 정규화 범위로 변환."""
    return 2.0 * (x - lower) / (upper - lower) - 1.0


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    """텐서 요소별 상/하한 클램핑."""
    return torch.max(torch.min(t, max_t), min_t)


def to_torch(x, dtype=torch.float, device: str | torch.device = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    """파이썬 리스트/넘파이 배열을 지정된 디바이스의 PyTorch 텐서로 변환."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
