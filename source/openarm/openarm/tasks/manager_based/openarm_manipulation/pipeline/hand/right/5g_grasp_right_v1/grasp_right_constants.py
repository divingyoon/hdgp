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
# 1D finger action: [-1=열림, +1=닫힘]
NUM_FINGER_ACTION = 1
# Total action dimensions
NUM_ACTIONS = NUM_PALM_POSE + NUM_FINGER_ACTION  # 7

# ---------------------------------------------------------------------------
# Hand Poses
# Joint order (20 DOF):
#   [0-3]   rj_dg_1_1~4  (thumb)
#   [4-7]   rj_dg_2_1~4  (index)
#   [8-11]  rj_dg_3_1~4  (middle)
#   [12-15] rj_dg_4_1~4  (ring)
#   [16-19] rj_dg_5_1~4  (pinky)
# ---------------------------------------------------------------------------

# 완전히 펼쳐진 자세 (action=-1 에 대응)
# 모든 관절 0: 손가락이 자연스럽게 펼쳐진 상태
HAND_OPEN_POSE = [
    0.0,  -1.5,  0.5,  0.5,   # thumb: neutral
    0.0,  0.0,  0.5,  0.5,   # index: neutral
    0.0,  0.0,  0.5,  0.5,   # middle: neutral
    0.0,  0.0,  0.5,  0.5,   # ring: neutral
    0.0,  0.0,  0.5,  0.5,   # pinky: neutral
]

# 컵 파지 자세 (action=+1 에 대응)
# 오른손 기준 (right hand):
#   thumb:  _1(X, 외전)=0, _2(Z, opposition)=-1.5 [한계 -π~0], _3=0.5, _4=0.5
#   index:  _1(X, 외전)=0, _2(Y, curl)=0.7 [한계 0~2.007], _3=0.5, _4=0.5
#   middle: _1(X, 외전)=0, _2(Y, curl)=0.7 [한계 0~1.955], _3=0.5, _4=0.5
#   ring:   _1(X, 외전)=0, _2(Y, curl)=0.7 [한계 0~1.902], _3=0.5, _4=0.5
#   pinky:  _1(Z, 굽힘)=0 [사용자 확인], _2(X, 외전)=0, _3=0.7, _4=0.5
HAND_GRASP_POSE = [
    0.0, -1.5,  0.5,  0.5,   # thumb: opposition curl
    0.0,  0.7,  0.5,  0.5,   # index: proximal curl
    0.0,  0.7,  0.5,  0.5,   # middle: proximal curl
    0.0,  0.7,  0.5,  0.5,   # ring: proximal curl
    0.0,  0.0,  0.7,  0.5,   # pinky: mid curl (_3=0.7)
]

# ---------------------------------------------------------------------------
# Arm Start Pose (OpenArm right arm)
# +Y 접근을 위한 초기 관절 시드 (시뮬 기준으로 튜닝 필요)
# palm 예상 위치 ≈ (0.42, -0.16, 0.53)
# ---------------------------------------------------------------------------
ARM_START_POSE = [1.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0]  

# ---------------------------------------------------------------------------
# Palm Pose Workspace (OpenArm right arm, +Y approach grasp)
# 회전 중심: ez=0°, ey=+90°, ex=0°
#   - palm_center local +x(= palm_link +Y, out-of-palm)가 물체 +Y를 바라봄
#   - 접근은 테이블 XY 평면과 평행한 -Y -> +Y 수평 이동
#   - palm_link +Z(손가락)는 수평 +X 방향
# ---------------------------------------------------------------------------
def PALM_POSE_MINS_FUNC(max_pose_angle: float) -> list:
    """max_pose_angle(deg): 회전 중심에서 ±로 허용되는 각도."""
    d = math.pi / 180.0
    return [
        0.25, -0.50, 0.20,                    # x, y, z [m]  (y_min=-0.50: 물체(y=-0.15)로부터 충분히 -Y)
        (0.0    - max_pose_angle) * d,         # ez: 0° ± angle
        (90.0   - max_pose_angle) * d,         # ey: +90° ± angle
        (0.0    - max_pose_angle) * d,         # ex: 0° ± angle
    ]


def PALM_POSE_MAXS_FUNC(max_pose_angle: float) -> list:
    d = math.pi / 180.0
    return [
        0.70, -0.05, 0.60,                    # y_max=-0.05: 물체(y=-0.15)보다 약간 안쪽까지 허용
        (0.0    + max_pose_angle) * d,
        (90.0   + max_pose_angle) * d,
        (0.0    + max_pose_angle) * d,
    ]


# ---------------------------------------------------------------------------
# Object & Goal
# ---------------------------------------------------------------------------
# 단일 컵 spawn 중심 위치 (오른팔 작업공간 중앙)
OBJECT_SPAWN_CENTER = [0.55, -0.15, 0.38]   # [x, y, z_base_of_cup] in world local frame
OBJECT_SPAWN_RANGE_XY = 0.08                # ±0.08m 무작위 오프셋

# 목표 위치: 컵을 들어 올릴 목표점
OBJECT_GOAL_POS = [0.55, -0.15, 0.65]       # z=0.65: 테이블 위 0.4m 들기

# ---------------------------------------------------------------------------
# Observation Size
# robot_dof_pos: 27, robot_dof_vel: 27
# hand_pos (6 bodies × 3D): 18
# object_pos: 3, object_rot: 4, goal_pos: 3
# last_actions: 7
# fabric_q: 27, fabric_qd: 27
# Total: 143
# ---------------------------------------------------------------------------
NUM_OBSERVATIONS = 143
