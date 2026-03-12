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

"""상수 정의: 5g_grasp_right_v3.

v3 변경점 (rl-icub-dexterous-manipulation 이식):
- num_fingers_touching 관측 추가 (1D): 이식 출처 iCub diff_num_contacts
- per_finger_touch_binary 관측 추가 (5D): 손가락별 접촉 여부, 이식 출처 iCub obs['touch']
- min_fingers_touching_for_success 성공 게이트
- scale_lift_reward_with_contacts 리프트 보상 스케일링
- 잔차 학습(RESPRECT) obs 슬롯 추가 (NUM_ACTIONS = 11D)
"""

from .grasp_right_preset import (
    HAND_GRASP_POSE,
    HAND_PCA_MAXS,
    HAND_PCA_MINS,
    HAND_START_POSE,
    OBJECT_GOAL_POS,
    OBJECT_SPAWN_CENTER,
    OBJECT_SPAWN_RANGE_XY,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_START_POSE,
    RIGHT_HAND_JOINT_NAMES,
    palm_pose_maxs,
    palm_pose_mins,
)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF = len(RIGHT_ARM_JOINT_NAMES)
NUM_HAND_DOF = len(RIGHT_HAND_JOINT_NAMES)
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF

NUM_PALM_POSE = 6
NUM_PCA_ACTION = 5
NUM_ACTIONS = NUM_PALM_POSE + NUM_PCA_ACTION
NUM_OBJECT_PC_POINTS = 32
NUM_OBJECT_PC_FEATURE = NUM_OBJECT_PC_POINTS * 3

# v1 호환 alias
ARM_START_POSE = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs

# ---------------------------------------------------------------------------
# Observation Size
# robot_dof_pos: 27, robot_dof_vel: 27
# hand_pos (7 bodies × 3D): 21
# object_pos: 3, object_rot: 4, goal_pos: 3
# object_init_pos: 3, pregrasp_delta: 3, pregrasp_target_x_dir: 3
# object pc feature: 32 * 3 = 96
# last_actions: 11
# fabric_q: 27, fabric_qd: 27
# num_fingers_touching (contact count, normalized): 1   ← v3 추가 (iCub 이식)
# per_finger_touch_binary (5D binary, per-finger): 5    ← v3 추가 (iCub obs['touch'] 이식)
# Total: 261
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 261  # v3: +1 num_fingers_norm, +5 per_finger_touch_binary
