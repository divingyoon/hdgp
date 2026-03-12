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

"""Hand/robot preset metadata for 5g_grasp_right_v3."""

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


# ---------------------------------------------------------------------------
# Direct PD hand control (v3: iCub-style joint control)
# ---------------------------------------------------------------------------

# RL이 직접 제어하는 curl joints (5D action, 손가락당 1D)
# thumb: rj_dg_1_2 (Z축, 음수=curl), 나머지: rj_dg_*_2 (Y축, 양수=curl)
# pinky: _1이 고정되어 있으므로 _3을 curl로 사용
HAND_CURL_JOINT_NAMES = [
    "rj_dg_1_2",  # thumb curl (Z, range [-π, 0])
    "rj_dg_2_2",  # index curl (Y, range [0, 2.007])
    "rj_dg_3_2",  # middle curl (Y, range [0, 1.955])
    "rj_dg_4_2",  # ring curl (Y, range [0, 1.902])
    "rj_dg_5_3",  # pinky curl (Y, range [-π/2, π/2]) — _1 고정으로 _3 사용
]

# 고정 joints (RL 제어 제외, 상수값 유지)
# rj_dg_1_1은 사용자 요청으로 제외(별도), 나머지 _1은 전부 고정
HAND_FIXED_JOINT_NAMES = [
    "rj_dg_1_1",  # thumb abduction (X): 0.0 고정
    "rj_dg_2_1",  # index abduction (X): 0.0 고정
    "rj_dg_3_1",  # middle abduction (X): 0.0 고정
    "rj_dg_4_1",  # ring abduction (X): 0.0 고정
    "rj_dg_5_1",  # pinky Z-flex: 0.0 고정
    "rj_dg_5_2",  # pinky abduction (X): 0.0 고정
]
HAND_FIXED_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# iCub distal tendon 커플링 (PIP = _3, DIP = _4)
# PIP joints: curl 기준으로 커플링
HAND_PIP_JOINT_NAMES = [
    "rj_dg_1_3",  # thumb PIP (X): abs(curl_1_2) * ratio
    "rj_dg_2_3",  # index PIP (Y): curl_2_2 * ratio
    "rj_dg_3_3",  # middle PIP (Y): curl_3_2 * ratio
    "rj_dg_4_3",  # ring PIP (Y): curl_4_2 * ratio
    "rj_dg_5_4",  # pinky DIP (Y): curl_5_3 * ratio (pinky는 _3이 curl이므로 _4가 커플링)
]
# DIP joints (thumb,index,middle,ring만 — pinky는 _4가 위의 PIP에 포함)
HAND_DIP_JOINT_NAMES = [
    "rj_dg_1_4",  # thumb DIP (X): abs(curl_1_2) * ratio
    "rj_dg_2_4",  # index DIP (Y): curl_2_2 * ratio
    "rj_dg_3_4",  # middle DIP (Y): curl_3_2 * ratio
    "rj_dg_4_4",  # ring DIP (Y): curl_4_2 * ratio
]

# iCub distal tendon 커플링 비율
# HAND_GRASP_POSE 기준: thumb(_2=-1.5,_3=0.5,_4=0.5), index(_2=0.7,_3=0.5,_4=0.5)
# → thumb ratio = 0.5/1.5 ≈ 0.33, index ratio = 0.5/0.7 ≈ 0.71
DISTAL_RATIO_PIP = [0.33, 0.71, 0.71, 0.71, 0.71]  # per finger (PIP = curl × ratio)
DISTAL_RATIO_DIP = [0.33, 0.71, 0.71, 0.71]          # per finger (DIP, thumb~ring only)

# curl joint 절대 범위 [min, max] (rad) — iCub ctrlrange에 해당
CURL_JOINT_LIMITS_MIN = [-_math.pi, 0.0,  0.0,  0.0,  0.0]
CURL_JOINT_LIMITS_MAX = [0.0, 2.007, 1.955, 1.902, _math.pi / 2]
