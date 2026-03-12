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

"""상수 정의: 5g_grasp_right_v4.

Observation 구조 (DEXTRAH distillation 방식 기준):
  Actor = Critic = 152D  →  separate: False, 공유 backbone

  Teacher (v4) obs 152D:
    joint_pos (27D) + joint_vel (27D)
    palm_center_pos (3D) + fingertip_pos (15D)   ← Fabrics FK
    fabric_q (27D) + fabric_qd (27D)              ← Fabrics 내부 상태 (real에서도 실행)
    cup_pos (3D) + cup_rot (4D)                   ← teacher: ground truth
    goal_pos (3D) + touch_binary (5D) + last_actions (11D)
    = 152D

  Student (나중) obs 152D:
    동일 구조, cup_pos/cup_rot 슬롯만 Foundation Pose 추정값으로 교체
    → 차원 동일 유지, teacher→student 이식 시 재설계 불필요

  DEXTRAH 참조:
    - Teacher/Critic 동일 obs → separate: False
    - Fabrics는 real robot에서도 실행 → fabric_q/qd는 privileged가 아님
    - Asymmetry는 teacher↔student 사이 (DAgger), actor↔critic 사이가 아님

Action (11D):
  [0:6] palm pose (x, y, z, ez, ey, ex), 정규화 [-1, 1]
  [6:11] finger curl Δq (5D), 정규화 [-1, 1]
"""

from .grasp_right_preset import (
    HAND_GRASP_POSE,
    HAND_START_POSE,
    OBJECT_GOAL_POS,
    OBJECT_SPAWN_CENTER,
    OBJECT_SPAWN_RANGE_XY,
    PREGRASP_OFFSET,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_START_POSE,
    RIGHT_HAND_JOINT_NAMES,
    palm_pose_maxs,
    palm_pose_mins,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF = len(RIGHT_ARM_JOINT_NAMES)     # 7
NUM_HAND_DOF = len(RIGHT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF   # 27

NUM_PALM_POSE = 6
NUM_CURL_ACTION = 5
NUM_ACTIONS = NUM_PALM_POSE + NUM_CURL_ACTION  # 11

NUM_FINGERTIPS = 5

# ---------------------------------------------------------------------------
# Obs (actor = critic = 동일): 152D
#   joint_pos: 27, joint_vel: 27
#   palm_center_pos: 3, fingertip_pos: 15
#   fabric_q: 27, fabric_qd: 27          ← Fabrics 내부 상태 (real에서도 실행)
#   cup_pos: 3, cup_rot: 4, goal_pos: 3
#   touch_binary: 5, last_actions: 11
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 152   # actor = critic, separate: False

# v1 호환 alias
ARM_START_POSE = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs
