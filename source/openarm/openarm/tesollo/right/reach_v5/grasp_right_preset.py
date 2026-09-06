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

"""로봇 및 손 프리셋 메타데이터: open-tesol_r_reach_v5

순수 접근(Pure Reach) 태스크 전용 경량 프리셋:
- 관절 그룹: 우팔 7-DOF (`r_aj_1~7`) + 우손 20-DOF (`r_hj_*`)
- 손가락 정적 포즈: `HAND_APPROACH_POSE` (엄지 opposition pre-curl, 4지 완전 개방)
- 작업 공간 (Workspace): 테이블 위 컵 스폰 좌표 및 Side-Approach 사전 오프셋
- 손바닥 가두리 박스: `palm_pose_mins()`, `palm_pose_maxs()`
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# 관절 그룹 정의 (Joint Groups)
# ---------------------------------------------------------------------------
# 통일 네이밍 (openarm_tesollo_bi_s_rl.usd 기준)
_R_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RIGHT_ARM_JOINT_NAMES = [f"r_aj_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINT_NAMES = [f"r_hj_{f}_{j}" for f in _R_FINGERS for j in range(1, 5)]
RIGHT_ACTUATED_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

LEFT_ARM_JOINT_NAMES = [f"l_aj_{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINT_NAMES = ["l_hj_gripper_1", "l_hj_gripper_2"]
LEFT_ARM_AND_GRIPPER_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES

# 유휴 좌팔 및 그리퍼 대기 자세 (sensor_rl 기준 0.044 개방 고정)
LEFT_ARM_REST_JOINT_POS = {
    "l_aj_1": 0.0,
    "l_aj_2": 0.0,
    "l_aj_3": 0.0,
    "l_aj_4": 0.0,
    "l_aj_5": 0.0,
    "l_aj_6": 0.0,
    "l_aj_7": 0.0,
    "l_hj_gripper_1": 0.044,
    "l_hj_gripper_2": 0.044,
}


# ---------------------------------------------------------------------------
# 바디 / 링크 네이밍 (Body / Link Names)
# ---------------------------------------------------------------------------
# Isaac Lab PhysX USD 기준 바디명
HAND_BODY_NAMES_USD = [
    "r_hl_palm",
    "r_hl_thumb_4",
    "r_hl_index_4",
    "r_hl_middle_4",
    "r_hl_ring_4",
    "r_hl_pinky_4",
]

# Fabrics IK 해석용 프레임 (openarm_tesollo fabrics URDF 기준)
FABRIC_HAND_BODY_NAMES = [
    "r_hl_palm",
    "r_hl_palm_x",  # 손바닥 +X 법선 벡터 참조 헬퍼
    "r_hl_thumb_tip",
    "r_hl_index_tip",
    "r_hl_middle_tip",
    "r_hl_ring_tip",
    "r_hl_pinky_tip",
]


# ---------------------------------------------------------------------------
# 로봇 및 손 포즈 프리셋 (Pose Presets)
# ---------------------------------------------------------------------------
# 우팔 시작 자세 (Q_REF 근처 테이블 위 안전 자세)
RIGHT_ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]

# 손가락 접근 준비 자세 (Reach 태스크에서 손가락이 고정 유지할 이상적인 포즈)
# 엄지(thumb)는 opposition 방향(-1.57 rad)으로 미리 꺾어 컵 충돌을 방지하고, 나머지 4지는 완전 개방
HAND_APPROACH_POSE = [
    0.0, -1.57, -0.5, 0.0,   # thumb: opposition curl
    0.0,  0.0,   0.0, 0.0,   # index: fully open
    0.0,  0.0,   0.0, 0.0,   # middle: fully open
    0.0,  0.0,   0.0, 0.0,   # ring: fully open
    0.0,  0.0,   0.0, 0.0,   # pinky: fully open
]

# 완전 개방 자세 (0.0 rad)
HAND_START_POSE = [0.0] * 20


# ---------------------------------------------------------------------------
# 작업 공간 및 타겟 지오메트리 (Workspace & Spawn Geometry)
# ---------------------------------------------------------------------------
# 컵 스폰 기준점 (로봇 베이스 로컬 좌표계 [X, Y, Z])
OBJECT_SPAWN_CENTER = [0.27, -0.10, 0.38]
OBJECT_SPAWN_RANGE_XY = 0.06  # ±6cm 범위 랜덤화

# 측면 접근(Side Approach) 사전 오프셋 (컵 중심 기준 Y축 -12cm, Z축 +5cm)
PREGRASP_OFFSET = [0.0, -0.12, 0.05]


# ---------------------------------------------------------------------------
# 손바닥 유효 작업 영역 바운딩 박스 (Palm Workspace Bounds)
# ---------------------------------------------------------------------------
def palm_pose_mins(max_pose_angle: float) -> list[float]:
    """손바닥 EE의 최소 허용 위치 [X, Y, Z] 및 최소 오일러 각도 [ez, ey, ex]"""
    d = math.pi / 180.0
    return [
        0.20, -0.55, 0.20,
        (90.0 - max_pose_angle) * d,
        (0.0 - max_pose_angle) * d,
        (90.0 - max_pose_angle) * d,
    ]


def palm_pose_maxs(max_pose_angle: float) -> list[float]:
    """손바닥 EE의 최대 허용 위치 [X, Y, Z] 및 최대 오일러 각도 [ez, ey, ex]"""
    d = math.pi / 180.0
    return [
        0.65, 0.22, 0.65,
        (90.0 + max_pose_angle) * d,
        (0.0 + max_pose_angle) * d,
        (90.0 + max_pose_angle) * d,
    ]
