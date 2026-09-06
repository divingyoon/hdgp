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

"""환경 설정: open-tesol_r_reach_v5

순수 접근(Pure Reach) 태스크 전용 Isaac Lab 환경 설정:
- 물리 시뮬레이션: 120Hz, 정책 주기: 60Hz (decimation=2)
- 로봇: OpenArm-Tesollo 양팔 모델 (openarm_tesollo_bi_s_rl.usd)
- 모터 게인: 팔 7-DoF (Stiffness 400.0, Damping 80.0)
- 타겟 물체: MultiAsset 8종 컵 자산
- 리셋: 고정 홈 포즈 (reset_from_fixed_home)
- 보상: 가우시안 도달 커널, 법선 정렬, 액션 스무딩
"""

from __future__ import annotations

import os as _os
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from openarm import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS, NUM_CRITIC_OBSERVATIONS
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)

# ---------------------------------------------------------------------------
# 자산 경로 설정 (Asset Paths)
# ---------------------------------------------------------------------------
_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")
_SDF_ASSET_ROOT = _os.path.join(_ASSETS_DIR, "cup")

# ---------------------------------------------------------------------------
# 타겟 컵 물체 구성 (8종 MultiAsset)
# ---------------------------------------------------------------------------
_BASE_OBJECT_MASS: float = 0.134

_ACTIVE_OBJECT_SPECS: tuple[dict, ...] = (
    {"id": "cup_big_s085", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (0.85, 0.85, 0.85), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s100", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.00, 1.00, 1.00), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s115", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.15, 1.15, 1.15), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s130", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.30, 1.30, 1.30), "mass": _BASE_OBJECT_MASS},
    {"id": "shaker_closed", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "shaker_closed_rl.usd"), "scale": (1.0, 1.0, 1.0), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s090", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (0.90, 0.90, 0.90), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s105", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.05, 1.05, 1.05), "mass": _BASE_OBJECT_MASS},
    {"id": "cup_big_s120", "usd_path": _os.path.join(_SDF_ASSET_ROOT, "cup_big_rl.usd"), "scale": (1.20, 1.20, 1.20), "mass": _BASE_OBJECT_MASS},
)
_ACTIVE_OBJECT_NAMES: tuple[str, ...] = tuple(_s["id"] for _s in _ACTIVE_OBJECT_SPECS)


def _object_usd_cfg(spec: dict) -> sim_utils.UsdFileCfg:
    _mass_props = sim_utils.MassPropertiesCfg(mass=float(spec["mass"])) if "mass" in spec else None
    return sim_utils.UsdFileCfg(
        usd_path=spec["usd_path"],
        activate_contact_sensors=False,
        scale=spec["scale"],
        mass_props=_mass_props,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=100.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )


_GRASP_OBJECT_SPAWN = sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[_object_usd_cfg(_s) for _s in _ACTIVE_OBJECT_SPECS],
    random_choice=False,
)


# ---------------------------------------------------------------------------
# 환경 설정 클래스 (GraspRightEnvCfg)
# ---------------------------------------------------------------------------
@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """open-tesol_r_reach_v5 환경 설정."""

    # 1. 시뮬레이션 및 에피소드 파라미터
    episode_length_s: float = 3.3333333333333335  # 200 steps @ 60Hz
    decimation: int = 2                            # 120Hz PhysX / 2 = 60Hz Policy
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = False

    # 2. 관측 및 액션 공간 차원
    observation_space: int = NUM_OBSERVATIONS          # 37D
    action_space: int = NUM_ACTIONS                   # 6D
    state_space: int = NUM_CRITIC_OBSERVATIONS         # 37D

    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_CRITIC_OBSERVATIONS

    # 3. Fabrics IK 플래너 설정
    use_hand_fabric: bool = False
    max_pose_angle: float = 45.0
    fabrics_max_objects_per_env: int = 8
    fabrics_damping_gain: float = 20.0

    # 4. 리셋 및 델타 액션 설정
    reset_from_fixed_home: bool = True
    reset_home_palm_pose: tuple = (0.28, -0.38, 0.42, 90.0, 0.0, 90.0)

    pregrasp_offset_x: float = 0.00
    pregrasp_offset_y: float = -0.12  # 컵 측면 12cm 이격
    pregrasp_offset_z: float = 0.05   # 컵 높이 +5cm
    cup_grasp_z_offset: float = 0.05  # 호환성 별칭

    palm_delta_xyz: tuple = (0.15, 0.35, 0.15)  # (±m, Y축은 홈 y=-0.38에서 컵 y=-0.10 도달 위해 0.35)
    palm_delta_rot_deg: float = 20.0            # ±20° 회전 자유도

    # 5. 관측 노이즈 (Domain Randomization)
    obs_noise_joint_pos: float = 0.01    # rad
    obs_noise_joint_vel: float = 0.05    # rad/s
    obs_noise_body_pos: float = 0.005    # m
    obs_noise_cup_pos: float = 0.015     # m

    # 6. Reach v3 직교 분리 접근(Approach) 및 정렬 보상 가중치
    standoff_target_dist: float = 0.08         # 컵 중심 기준 수평 8cm Standoff 목표 (표면 앞 약 3.5cm)
    approach_xy_weight: float = 0.30           # XY 수평 오차 선형 페널티 (원거리 끌어당김)
    approach_xy_fine_weight: float = 0.35      # XY 수평 Tanh 정밀 도달 보상
    approach_xy_fine_std: float = 0.05         # XY Tanh 폭 (5cm)
    approach_z_weight: float = 0.30            # Z 높이 오차 선형 페널티
    approach_z_fine_weight: float = 0.35       # Z 높이 Tanh 정밀 일치 보상
    approach_z_fine_std: float = 0.03          # Z Tanh 폭 (3cm)
    approach_align_weight: float = 0.20        # 손바닥 피부 정면(+X) 컵 상대 대면 보상
    approach_down_align_weight: float = 0.10   # 4손가락(+Z) 하향 파지세 보상

    # 컵 외란 감점 (안정적 무충돌 접근 유도)
    cup_lin_vel_penalty_weight: float = 2.0    # 컵 선속도 감점
    cup_ang_vel_penalty_weight: float = 0.5    # 컵 각속도 감점
    approach_xy_penalty_weight: float = 5.0    # 컵 XY 밀림(변위) 감점
    approach_tilt_penalty_weight: float = 0.08 # 컵 기울어짐 감점
    tilt_penalty_margin_deg: float = 2.0       # 기울어짐 허용 마진(deg)

    # 행동 스무딩 및 목표 도달 보너스
    action_smooth_weight: float = -0.01        # 액션 크기 페널티
    action_rate_weight: float = -0.02          # 급격한 가속/떨림 페널티
    joint_vel_weight: float = -0.005           # 고속 관절 회전 억제
    reach_success_bonus: float = 5.0           # 목표 도달 성공 보너스


    # 7. 타겟 스폰 및 작업 공간
    object_spawn_x_center: float = 0.27
    object_spawn_y_center: float = -0.10
    object_spawn_z: float = 0.297
    object_spawn_x_range: float = 0.06
    object_spawn_y_range: float = 0.06

    active_object_names: tuple[str, ...] = _ACTIVE_OBJECT_NAMES
    object_bbox_path: str = _os.path.join(_ASSETS_DIR, "object_bbox.json")

    # 8. PhysX 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    # 9. 씬(Scene) 엔티티 구성
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        replicate_physics=False,
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.2],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                "r_aj_1":  0.5,
                "r_aj_2":  0.1,
                "r_aj_3":  0.4,
                "r_aj_4":  0.60,
                "r_aj_5": -0.2,
                "r_aj_6":  0.0,
                "r_aj_7":  0.0,
                "r_hj_thumb_1": 0.0, "r_hj_thumb_2": -1.57, "r_hj_thumb_3": -0.5, "r_hj_thumb_4": 0.0,
                "r_hj_index_1": 0.0, "r_hj_index_2":  0.0,  "r_hj_index_3":  0.0, "r_hj_index_4": 0.0,
                "r_hj_middle_1": 0.0, "r_hj_middle_2":  0.0,  "r_hj_middle_3":  0.0, "r_hj_middle_4": 0.0,
                "r_hj_ring_1": 0.0, "r_hj_ring_2":  0.0,  "r_hj_ring_3":  0.0, "r_hj_ring_4": 0.0,
                "r_hj_pinky_1": 0.0, "r_hj_pinky_2":  0.0,  "r_hj_pinky_3":  0.0, "r_hj_pinky_4": 0.0,
                **LEFT_ARM_REST_JOINT_POS,
            },
        ),
        actuators={
            "head_camera": ImplicitActuatorCfg(
                joint_names_expr=["head_j_(pan|tilt)"],
                stiffness=400.0,
                damping=80.0,
            ),
            "right_arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[1-3]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.213,
            ),
            "right_arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_4"],
                stiffness=400.0,
                damping=80.0,
                friction=0.493,
            ),
            "right_arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["r_aj_[5-7]"],
                stiffness=400.0,
                damping=80.0,
                friction=0.151,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["l_aj_[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "tesollo_right_hand": ImplicitActuatorCfg(
                joint_names_expr=["r_hj_[a-z]+_[1-4]"],
                stiffness=400.0,
                damping=60.0,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["l_hj_gripper_[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.25],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=_GRASP_OBJECT_SPAWN,
    )

    hand_body_names: list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
