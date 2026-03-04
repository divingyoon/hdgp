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

"""환경 설정: 5g_grasp_right_v1

OpenArm(7 DOF) + Teosllo(20 DOF) 오른손 단일 컵 파지 태스크.
- 제어 방식: Geometric Fabrics (DEXTRAH 방식)
- 로봇 USD: openarm_modular_dual (Teosllo 공식 물성치 반영)
- 액션: 7D = 6D palm pose + 1D finger interpolation
- 리워드: KUKA_ALLEGRO 방식 4항목
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

import os as _os

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS

# hdgp 루트 및 assets 경로 (portable)
_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v1 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    episode_length_s: float = 10.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = True

    # -----------------------------------------------------------------------
    # 관측·액션 공간 (DirectRLEnvCfg 필수 필드)
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS  # 143
    action_space: int = NUM_ACTIONS            # 7
    state_space: int = NUM_OBSERVATIONS        # critic obs = policy obs (단순화)

    # 내부 참조용 (DEXTRAH 호환)
    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    # Palm pose 워크스페이스 각도 한계 (deg): 회전 중심 ±
    max_pose_angle: float = 45.0

    # -----------------------------------------------------------------------
    # 리워드 파라미터 (KUKA_ALLEGRO 방식)
    # -----------------------------------------------------------------------
    # 1. hand_to_object: exp(-α * max_dist) — 손 전체가 물체에 접근하도록
    hand_to_object_weight: float = 1.0
    hand_to_object_sharpness: float = 10.0   # threshold ≈ 1/10 = 10 cm

    # 2. object_to_goal: exp(-α * dist) — 물체를 목표 위치로
    object_to_goal_weight: float = 5.0
    object_to_goal_sharpness: float = 15.0   # threshold ≈ 6.7 cm

    # 3. lift: exp(-α * |z_obj - z_goal|) — 수직 들기
    lift_weight: float = 5.0
    lift_sharpness: float = 5.0              # baseline at 0.50 m above table

    # 4. finger_curl_reg: w * ||q_hand - q_grasp||^2 (패널티)
    finger_curl_weight: float = -0.01

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.55
    object_spawn_y_center: float = -0.15
    object_spawn_z: float = 0.38        # visdex 물체 z_min=-0.076 기준 안전 높이
    object_spawn_xy_range: float = 0.08  # ±0.08m 균등 분포

    # 성공 기준: 물체가 goal 0.1m 이내
    success_threshold: float = 0.10

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=64 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_rigid_contact_count=2**23,
            gpu_collision_stack_size=2**23,
            gpu_max_num_partitions=8,
            friction_correlation_distance=0.00625,
        ),
    )

    # -----------------------------------------------------------------------
    # 씬 설정
    # -----------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -----------------------------------------------------------------------
    # 테이블 설정 (kinematic=True → 고정, 물리적 충돌만 처리)
    # robot pos=[0,0,-0.25] → table pos=[0.5725, 0.003, -0.25] → table top ≈ z=0.13m
    # cup spawn_z=0.38은 table top(0.13) + 컵 높이(0.25) 기준
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.235],   # DEXTRAH table 기준: top ≈ z=0.25m
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # 고정 테이블 (물리 반응 없음)
                disable_gravity=True,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇 설정
    # openarm_modular_dual.usd:
    #   - OpenArm bimanual + Teosllo right hand
    #   - Teosllo 공식 물성치(stiffness/damping)가 USD에 반영됨
    #   - OpenArm arm: Isaac Lab에서 stiffness/damping 설정 (정밀 제어)
    #   - Teosllo hand: USD 물성치 유지 (ImplicitActuatorCfg에서 override 안 함)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_modular_dual/openarm_modular_dual.usd"),
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
                # 오른팔: 자연 자세 (FK → palm ≈ [0.418, -0.157, 0.534])
                "openarm_right_joint1":  1.0,
                "openarm_right_joint2": -0.1,
                "openarm_right_joint3":  0.0,
                "openarm_right_joint4":  0.5,
                "openarm_right_joint5":  0.0,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                # 오른손: 컵 파지 자세 (grasp pose)
                "rj_dg_1_1": 0.0, "rj_dg_1_2": -1.5, "rj_dg_1_3": 0.5, "rj_dg_1_4": 0.5,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.7, "rj_dg_2_3": 0.5, "rj_dg_2_4": 0.5,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.7, "rj_dg_3_3": 0.5, "rj_dg_3_4": 0.5,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.7, "rj_dg_4_3": 0.5, "rj_dg_4_4": 0.5,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0, "rj_dg_5_3": 0.7, "rj_dg_5_4": 0.5,
                # 왼팔: 학습 제외, 고정 자세
                "openarm_left_joint1": -0.5,
                "openarm_left_joint2": -0.5,
                "openarm_left_joint3":  0.6,
                "openarm_left_joint4":  0.7,
                "openarm_left_joint5":  0.0,
                "openarm_left_joint6":  0.0,
                "openarm_left_joint7": -1.0,
                # 왼쪽 그리퍼: 0 (openarm_modular_dual.usd 기준 — Teosllo 없음)
                "openarm_left_finger_joint1": 0.0,
                "openarm_left_finger_joint2": 0.0,
            },
        ),
        actuators={
            # OpenArm arm: Isaac Lab에서 stiffness/damping 설정
            "openarm_right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            # Teosllo right hand: USD 공식 물성치 유지 (stiffness/damping = None → USD 값)
            "tesollo_right_hand": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_.*"],
                stiffness=None,
                damping=None,
            ),
            # 왼쪽 그리퍼: 고정
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # 컵 설정 (단일 컵 파지)
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, -0.15, 0.38],   # 테이블 위 초기 위치 (DEXTRAH table top≈0.25 + 여유)
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup_bead/cup.usd"),
            scale=(1.0, 1.0, 1.2),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 머그 설정 (mug: 초기 위치는 scene 밖, reset 시 랜덤 배정)
    # -----------------------------------------------------------------------
    mug_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -10.0],   # 초기에는 scene 밖 아래에 배치
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup_bead/mug.usd"),
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # Hand body names (Isaac Sim USD 기준)
    # rj_dg_palm: rj_dg_palm joint를 revolute(range=0)으로 변경 → 별도 body 등록
    # -----------------------------------------------------------------------
    hand_body_names: list = [
        "rl_dg_palm",    # 손바닥 (palm center proxy)
        "rl_dg_1_4",     # thumb tip
        "rl_dg_2_4",     # index tip
        "rl_dg_3_4",     # middle tip
        "rl_dg_4_4",     # ring tip
        "rl_dg_5_4",     # pinky tip
    ]

    # -----------------------------------------------------------------------
    # Actuated joint names (오른팔 + 오른손, 27 DOF)
    # -----------------------------------------------------------------------
    actuated_joint_names: list = (
        [f"openarm_right_joint{i}" for i in range(1, 8)] +  # arm 7 DOF
        [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]  # hand 20 DOF
    )

    # 왼팔+왼쪽 그리퍼 joint names (고정, openarm_modular_dual.usd 기준)
    left_arm_joint_names: list = (
        [f"openarm_left_joint{i}" for i in range(1, 8)] +
        ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
    )
