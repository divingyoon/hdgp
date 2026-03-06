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

"""상수 정의: 5g_grasp_right_v2.

v2 1단계 목표는 hand/robot 메타데이터를 preset 모듈로 분리해
태스크 코드의 하드코딩 중복을 줄이는 것이다.
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
# Total: 255
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 255
