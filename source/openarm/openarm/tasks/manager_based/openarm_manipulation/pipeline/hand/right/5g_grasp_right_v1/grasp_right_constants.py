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

"""상수 정의: 5g_grasp_right_v1

OpenArm(7 DOF) + Teosllo(20 DOF) 단일 컵 파지 태스크.
PCA 대신 2가지 수동 정의 포즈를 선형 보간으로 사용.
"""

import math

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF = 7    # OpenArm right arm (openarm_right_joint1~7)
NUM_HAND_DOF = 20  # Teosllo right hand (rj_dg_1_1~4 ... rj_dg_5_1~4)
NUM_ROBOT_DOF = NUM_ARM_DOF + NUM_HAND_DOF  # 27

# 6D palm pose action: [x, y, z, ez, ey, ex]
NUM_PALM_POSE = 6
# 손가락은 palm_dist 기반 자동 보간 (action 없음)
NUM_PCA_ACTION = 0
# Total action dimensions
NUM_ACTIONS = NUM_PALM_POSE  # 6

# ---------------------------------------------------------------------------
# Hand Poses (DEXTRAH 방식)
# Joint order (20 DOF):
#   [0-3]   rj_dg_1_1~4  (thumb)
#   [4-7]   rj_dg_2_1~4  (index)
#   [8-11]  rj_dg_3_1~4  (middle)
#   [12-15] rj_dg_4_1~4  (ring)
#   [16-19] rj_dg_5_1~4  (pinky)
# ---------------------------------------------------------------------------

# 리셋 시작 자세: 손가락 열린 상태 (컵 소환 영역 관통 방지)
# thumb _2=0: opposition 없음 → 엄지가 컵 소환 범위 밖에 위치
HAND_START_POSE = [
    0.0,  0.0,  0.0,  0.0,   # thumb: fully open
    0.0,  0.0,  0.0,  0.0,   # index: fully open
    0.0,  0.0,  0.0,  0.0,   # middle: fully open
    0.0,  0.0,  0.0,  0.0,   # ring: fully open
    0.0,  0.0,  0.0,  0.0,   # pinky: fully open
]

# Fabrics cspace attractor 목표 자세 (DEXTRAH curled_q에 해당)
# 에이전트가 PCA로 손가락을 조정하는 기준점
# GUI로 직접 확인한 자연스러운 파지 자세
#   thumb:  _1=0, _2=-1.5 (opposition), _3=0.5, _4=0.5
#   index:  _1=0, _2=0.7 (curl),        _3=0.5, _4=0.5
#   middle: _1=0, _2=0.7,               _3=0.5, _4=0.5
#   ring:   _1=0, _2=0.7,               _3=0.5, _4=0.5
#   pinky:  _1=0, _2=0,   _3=0.7,       _4=0.5
HAND_GRASP_POSE = [
    0.0, -1.5,  0.5,  0.5,   # thumb
    0.0,  0.7,  0.5,  0.5,   # index
    0.0,  0.7,  0.5,  0.5,   # middle
    0.0,  0.7,  0.5,  0.5,   # ring
    0.0,  0.0,  0.7,  0.5,   # pinky
]

# ---------------------------------------------------------------------------
# Arm Start Pose (OpenArm right arm)
# FK 탐색 결과 (j7=0, Fabrics default_config와 일치):
#   palm_center ≈ (0.463, -0.311, 0.431)
#   palm_normal_y ≈ 0.975 (+Y 방향, 컵 접근 방향)
#   컵 spawn(y=-0.15) 기준 0.161m 뒤에서 시작 → +Y 방향으로 직진하여 파지
# 기존 [1.0,-0.5,0,0.5,0,0,1.571]: j7=1.571 때문에 palm이 y=-0.01에 위치
#   → 컵보다 앞쪽(+y)에 있어 접근 방향이 역방향, 엄지가 앞으로 돌출
# ---------------------------------------------------------------------------
ARM_START_POSE = [0.5, 0.1, 0.4, 0.8, -0.2, 0.0, 0.0]

# ---------------------------------------------------------------------------
# Palm Pose Workspace (OpenArm right arm, +Y approach grasp)
# 회전 중심: ez=+90°, ey=0°, ex=+90°
#   - palm_link +X(손바닥 법선) = world +Y
#   - 접근은 테이블 XY 평면과 평행한 -Y -> +Y 수평 이동
#   - palm_link +Z(손가락)는 수평 +X 방향
# ---------------------------------------------------------------------------
def PALM_POSE_MINS_FUNC(max_pose_angle: float) -> list:
    """max_pose_angle(deg): 회전 중심에서 ±로 허용되는 각도."""
    d = math.pi / 180.0
    return [
        0.25, -0.50, 0.20,                    # x, y, z [m]  (y_min=-0.50: 물체(y=-0.15)로부터 충분히 -Y)
        (90.0   - max_pose_angle) * d,         # ez: +90° ± angle
        (0.0    - max_pose_angle) * d,         # ey:   0° ± angle
        (90.0   - max_pose_angle) * d,         # ex: +90° ± angle
    ]


def PALM_POSE_MAXS_FUNC(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.70, -0.05, 0.60,                    # y_max=-0.05: 물체(y=-0.15)보다 약간 안쪽까지 허용
        (90.0   + max_pose_angle) * d,
        (0.0    + max_pose_angle) * d,
        (90.0   + max_pose_angle) * d,
    ]


# ---------------------------------------------------------------------------
# Object & Goal
# ---------------------------------------------------------------------------
# 단일 컵 spawn 중심 위치 (오른팔 작업공간 중앙)
OBJECT_SPAWN_CENTER = [0.55, -0.15, 0.30]   # [x, y, z] 테이블 top(0.25m) + 0.05m 여유
OBJECT_SPAWN_RANGE_XY = 0.08                # ±0.08m 무작위 오프셋

# 목표 위치: 컵을 들어 올릴 목표점
OBJECT_GOAL_POS = [0.55, -0.15, 0.65]       # z=0.65: 테이블 위 0.4m 들기

# ---------------------------------------------------------------------------
# Observation Size
# robot_dof_pos: 27, robot_dof_vel: 27
# hand_pos (7 bodies × 3D): 21  ← palm_center + palm_x + 5 tips
# object_pos: 3, object_rot: 4, goal_pos: 3
# last_actions: 6  ← 6D palm only (손가락은 자동 보간)
# fabric_q: 27, fabric_qd: 27
# Total: 145
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 145
