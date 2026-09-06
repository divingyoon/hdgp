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

"""상수 정의: open-tesol_r_reach_v5

순수 접근(Pure Reach) 태스크를 위한 경량화 클린 상수 정의:
- Action (6D): 6D Palm Pose Target (ΔX, ΔY, ΔZ, ΔRoll, ΔPitch, ΔYaw) → Fabrics IK → Arm 7-DOF
- Hand: 20-DOF 손가락은 HAND_APPROACH_POSE 로 고정 (Action 0D)
- Observation (37D):
    arm_joint_pos:       7
    arm_joint_vel:       7
    palm_pos_w:          3
    palm_quat_w:         4
    target_cup_pos_w:    3
    target_cup_quat_w:   4
    palm_to_cup_rel_pos: 3
    last_actions:        6
    Total:              37
- Episode: 200 steps @ 60Hz (3.33s)
"""

from __future__ import annotations

import math

from .grasp_right_preset import (
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_ARM_START_POSE,
    palm_pose_mins,
    palm_pose_maxs,
)

# ---------------------------------------------------------------------------
# 로봇 자유도 (Robot Dimensions)
# ---------------------------------------------------------------------------
NUM_ARM_DOF: int = len(RIGHT_ARM_JOINT_NAMES)    # 우팔 7-DOF
NUM_HAND_DOF: int = len(RIGHT_HAND_JOINT_NAMES)  # 우손 20-DOF
NUM_ROBOT_DOF: int = NUM_ARM_DOF + NUM_HAND_DOF  # 총 27-DOF
NUM_FINGERTIPS: int = 5

# ---------------------------------------------------------------------------
# 액션 공간 (Action Space - Pure Reach 6D)
# ---------------------------------------------------------------------------
NUM_PALM_ACTION: int = 6  # 6D Palm Delta Pose (Fabrics IK 구동용)
NUM_ACTIONS: int = NUM_PALM_ACTION  # 6D

# ---------------------------------------------------------------------------
# 관측 공간 (Observation Space - 37D)
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS: int = 37
NUM_CRITIC_OBSERVATIONS: int = NUM_OBSERVATIONS  # 대칭 Actor-Critic 구조
NUM_STATES: int = NUM_OBSERVATIONS

# ---------------------------------------------------------------------------
# 에피소드 구조 (@ 60 Hz)
# ---------------------------------------------------------------------------
EPISODE_STEPS: int = 200  # 3.33초 (순수 접근 및 정렬 완료에 충분한 시간)

# ---------------------------------------------------------------------------
# 도달 성공 판정 임계값 (Reach Success Thresholds)
# ---------------------------------------------------------------------------
REACH_SUCCESS_XY_THRESHOLD: float = 0.02         # XY 수평 Standoff 오차 2cm 이내
REACH_SUCCESS_Z_THRESHOLD: float = 0.02          # Z 높이 오차 2cm 이내
REACH_ALIGNMENT_THRESHOLD_DEG: float = 25.0      # 손바닥 법선 정렬 오차 25도 이내 (cos ≈ 0.90)
REACH_SUCCESS_HOLD_STEPS: int = 10               # 10스텝 연속 유지 시 성공 판정


# ---------------------------------------------------------------------------
# 별칭 (Aliases)
# ---------------------------------------------------------------------------
ARM_START_POSE = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
