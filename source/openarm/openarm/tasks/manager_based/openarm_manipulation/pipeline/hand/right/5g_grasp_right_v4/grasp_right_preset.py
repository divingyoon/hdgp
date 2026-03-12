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

"""Hand/robot preset metadata for 5g_grasp_right_v4.

v3와 동일한 joint/body 구성. v4에서 재사용.
"""

import math
import math as _math


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

# Fabrics FK taskmap body names (openarm_tesollo.urdf 기준)
# [0]=palm_center, [1]=palm_x, [2:7]=fingertip ×5
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
# Start / grasp poses
# ---------------------------------------------------------------------------
# 완전히 열린 자세
# 컵 관통 방지는 리셋 순서로 해결 (컵 spawn을 Fabrics pregrasp 이후로)
HAND_START_POSE = [
    0.0, 0.0, 0.0, 0.0,   # thumb
    0.0, 0.0, 0.0, 0.0,   # index
    0.0, 0.0, 0.0, 0.0,   # middle
    0.0, 0.0, 0.0, 0.0,   # ring
    0.0, 0.0, 0.0, 0.0,   # pinky
]

# 파지 자세 (cspace attractor 기준)
HAND_GRASP_POSE = [
    0.0, -1.5, 0.5, 0.5,   # thumb
    0.0,  0.7, 0.5, 0.5,   # index
    0.0,  0.7, 0.5, 0.5,   # middle
    0.0,  0.7, 0.5, 0.5,   # ring
    0.0,  0.0, 0.7, 0.5,   # pinky
]

# 팔 시작 자세 (FK: palm ≈ [0.463, -0.311, 0.431])
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.8, -0.2, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Workspace / goal
# ---------------------------------------------------------------------------
# cup spawn center (local frame)
OBJECT_SPAWN_CENTER = [0.40, -0.15, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06
OBJECT_GOAL_POS = [0.40, -0.15, 0.65]

# Pregrasp offset: cup 옆(-Y 방향)에서 접근
# cup_pos + [0, -0.20, 0.05] → 컵 옆 20cm, 위로 5cm
# -0.12 → -0.20: palm Y=-0.35, 엄지 tip Y≈-0.23 (cup edge -0.19보다 4cm 여유)
PREGRASP_OFFSET = [0.0, -0.20, 0.05]


def palm_pose_mins(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.20, -0.55, 0.20,
        (90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.65, -0.02, 0.65,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]


# ---------------------------------------------------------------------------
# Direct PD hand control (v4: iCub-style, curl_gate 제거)
# ---------------------------------------------------------------------------

# RL이 직접 제어하는 curl joints (5D action, 손가락당 1D)
HAND_CURL_JOINT_NAMES = [
    "rj_dg_1_2",  # thumb curl (Z, range [-π, 0])
    "rj_dg_2_2",  # index curl (Y, range [0, 2.007])
    "rj_dg_3_2",  # middle curl (Y, range [0, 1.955])
    "rj_dg_4_2",  # ring curl (Y, range [0, 1.902])
    "rj_dg_5_3",  # pinky curl (Y, _1 고정이므로 _3 사용)
]

# 고정 joints (RL 제어 제외)
HAND_FIXED_JOINT_NAMES = [
    "rj_dg_1_1",  # thumb abduction: 0.0 고정
    "rj_dg_2_1",  # index abduction: 0.0 고정
    "rj_dg_3_1",  # middle abduction: 0.0 고정
    "rj_dg_4_1",  # ring abduction: 0.0 고정
    "rj_dg_5_1",  # pinky Z-flex: 0.0 고정
    "rj_dg_5_2",  # pinky abduction: 0.0 고정
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# iCub distal tendon 커플링 (PIP = _3, DIP = _4)
HAND_PIP_JOINT_NAMES = [
    "rj_dg_1_3",  # thumb PIP
    "rj_dg_2_3",  # index PIP
    "rj_dg_3_3",  # middle PIP
    "rj_dg_4_3",  # ring PIP
    "rj_dg_5_4",  # pinky DIP (pinky _3이 curl이므로 _4가 커플링)
]
HAND_DIP_JOINT_NAMES = [
    "rj_dg_1_4",  # thumb DIP
    "rj_dg_2_4",  # index DIP
    "rj_dg_3_4",  # middle DIP
    "rj_dg_4_4",  # ring DIP
]

# 커플링 비율 (HAND_GRASP_POSE 기준)
DISTAL_RATIO_PIP = [0.33, 0.71, 0.71, 0.71, 0.71]
DISTAL_RATIO_DIP = [0.33, 0.71, 0.71, 0.71]

# curl joint 절대 범위 [min, max] (rad)
CURL_JOINT_LIMITS_MIN = [-_math.pi, 0.0,  0.0,  0.0,  0.0]
CURL_JOINT_LIMITS_MAX = [0.0, 2.007, 1.955, 1.902, _math.pi / 2]
