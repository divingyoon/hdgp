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

"""Hand/robot preset metadata for 5g_grasp_right_v2."""

import math


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------
RIGHT_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT_NAMES = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

LEFT_ARM_REST_JOINT_POS = {
    "openarm_left_joint1": -0.5,
    "openarm_left_joint2": -0.5,
    "openarm_left_joint3": 0.6,
    "openarm_left_joint4": 0.7,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": -1.0,
    "openarm_left_finger_joint1": 0.0,
    "openarm_left_finger_joint2": 0.0,
}


# ---------------------------------------------------------------------------
# Hand links (USD / Fabrics)
# ---------------------------------------------------------------------------
HAND_BODY_NAMES_USD = [
    "rl_dg_palm",
    "rl_dg_1_4",
    "rl_dg_2_4",
    "rl_dg_3_4",
    "rl_dg_4_4",
    "rl_dg_5_4",
]

FABRIC_HAND_BODY_NAMES = [
    "palm_center",
    "palm_x",
    "tesollo_right_rl_dg_1_4",
    "tesollo_right_rl_dg_2_4",
    "tesollo_right_rl_dg_3_4",
    "tesollo_right_rl_dg_4_4",
    "tesollo_right_rl_dg_5_4",
]


# ---------------------------------------------------------------------------
# Start / grasp poses and PCA bounds
# ---------------------------------------------------------------------------
HAND_START_POSE = [
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
]

HAND_GRASP_POSE = [
    0.0, -1.5, 0.5, 0.5,
    0.0, 0.7, 0.5, 0.5,
    0.0, 0.7, 0.5, 0.5,
    0.0, 0.7, 0.5, 0.5,
    0.0, 0.0, 0.7, 0.5,
]

HAND_PCA_MINS = [0.0, -0.5, -1.0, -1.2, -0.5]
HAND_PCA_MAXS = [3.5, 2.0, 1.0, 2.0, 2.0]

RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.8, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace and object presets
# ---------------------------------------------------------------------------
OBJECT_SPAWN_CENTER = [0.55, -0.15, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.08
OBJECT_GOAL_POS = [0.55, -0.15, 0.65]


def palm_pose_mins(max_pose_angle: float) -> list[float]:
    d = math.pi / 180.0
    return [
        0.25, -0.50, 0.20,
        (90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list[float]:
    d = math.pi / 180.0
    return [
        0.70, -0.05, 0.60,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]
