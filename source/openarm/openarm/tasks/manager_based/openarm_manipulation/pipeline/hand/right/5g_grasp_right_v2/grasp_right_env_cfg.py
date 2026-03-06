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

"""환경 설정: 5g_grasp_right_v2

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import os as _os

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_constants import NUM_OBSERVATIONS, NUM_ACTIONS
from .grasp_right_preset import (
    HAND_BODY_NAMES_USD,
    LEFT_ARM_AND_GRIPPER_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_ACTUATED_JOINT_NAMES,
)

# hdgp 루트 및 assets 경로 (portable)
_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v2 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    # 물리: 120 Hz, 정책: 60 Hz (decimation=2)
    # Fabrics: fabrics_dt=1/60 × fabric_decimation=2 → 120 Hz
    episode_length_s: float = 10.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간 (DirectRLEnvCfg 필수 필드)
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS  # 255 (base 159 + object pc 96)
    action_space: int = NUM_ACTIONS            # 11 (6D palm + 5D PCA)
    state_space: int = NUM_OBSERVATIONS        # critic obs = policy obs (단순화)

    # 내부 참조용 (DEXTRAH 호환)
    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Object point-cloud feature observation (Phase B)
    # -----------------------------------------------------------------------
    use_object_pc_feature: bool = True
    object_pc_num_points: int = 32
    object_pc_feature_scale: float = 0.25
    object_pc_feature_clip: float = 4.0
    object_pc_nan_guard: bool = True

    # -----------------------------------------------------------------------
    # Object code -> point-cloud feature map (Phase C)
    # -----------------------------------------------------------------------
    # DemoGrasp 포팅용 파일 기반 object feature 관측 토글
    use_object_pc_feat: bool = True
    # object code feature 매핑(.pt/.npy) 파일 경로
    object_pc_feat_path: str = _os.path.join(
        _ASSETS_DIR,
        "object_pc_features/openarm_right_object_code_feat_dim64.pt",
    )
    # 파일 feature dim(관측 concat 시 고정 차원)
    object_pc_feat_dim: int = 64

    def __post_init__(self) -> None:
        """Recompute observation/state dimensions from active feature toggles."""
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        extra_obs = self.object_pc_feat_dim if self.use_object_pc_feat else 0
        total_obs = NUM_OBSERVATIONS + extra_obs
        self.observation_space = total_obs
        self.state_space = total_obs
        self.num_observations = total_obs
        self.num_states = total_obs

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    # Palm pose 워크스페이스 각도 한계 (deg): 회전 중심 ±
    max_pose_angle: float = 45.0
    # Fabrics world에서 env당 보유할 최대 장애물 슬롯 수
    # (활성 장애물 수보다 크게만 잡으면 됨. 과대 설정 시 메모리 낭비)
    fabrics_max_objects_per_env: int = 6

    # -----------------------------------------------------------------------
    # 리워드 파라미터
    # -----------------------------------------------------------------------
    # 1. hand_to_object (palm_center only): palm이 컵에 물리적으로 가까워야 보상
    #    sharpness=8 → palm_dist=0.08m: exp(-0.64)=0.53, 0.04m: exp(-0.32)=0.73
    #    mean(6 body) 대비: 등쪽 접근으로 게임 불가, palm 방향성 자연히 강제
    hand_to_object_weight: float = 2.0
    hand_to_object_sharpness: float = 8.0

    # 2. object_to_goal: exp(-α * dist) — 물체를 목표 위치로 (파지 후 단계)
    object_to_goal_weight: float = 5.0
    object_to_goal_sharpness: float = 15.0

    # 3. lift: exp(-α * |z_obj - z_goal|) — 수직 들기 (파지 후 단계)
    lift_weight: float = 5.0
    lift_sharpness: float = 5.0

    # 4. finger reward (proximity-gated, palm_dist 기준)
    # 기존 finger_curl_weight 단일 음수 방식 → 로컬 미니멈:
    #   palm_dist=0.05m에서 proximity=0.37 < 0.5 → 열린 자세가 더 유리
    # 해결: 두 항 분리
    #   finger_grasp_reward (+): 가까울 때 grasp → 양수 보상 (h2o 수준)
    #   finger_open_reg (-):     멀 때 닫으면 패널티 (조기 닫힘 방지)
    #
    # curl_proximity_sharpness=20 → palm_dist=0.05m: proximity=0.37, 0.035m: 0.50
    curl_proximity_sharpness: float = 20.0

    # grasp 보상: proximity * exp(-sharpness * grasp_error)
    # sharpness=1.0, mean 기준 grasp_error ≈ 0.3 (20 DOF 평균)
    # weight=2.0: h2o(weight=2.0)와 동급 → 명확한 유인
    finger_grasp_weight: float = 2.0
    finger_grasp_sharpness: float = 1.0

    # open 패널티: 멀 때 닫으면 페널티 (값 작게 유지)
    finger_open_weight: float = -0.02

    # 5. palm orientation reward: palm +X(손바닥 법선)이 컵을 향할수록 보상
    # align ∈ [-1, 1] → weight * align ∈ [-weight, +weight]
    # 위에서 덮기(palm +X = world -Z): align ≈ -1 → 패널티
    # 옆면 접근(palm +X = palm→cup 방향): align ≈ +1 → 보상
    palm_orient_weight: float = 1.0

    # -----------------------------------------------------------------------
    # 단계 전환 바이너리 게이트 파라미터
    # -----------------------------------------------------------------------
    # λ (approach_trigger): palm_dist < approach_trigger_dist → 파지 단계 시작
    #   0.12m: 컵 직경(≈0.08m)보다 약간 크게 → 손이 컵 옆면 근처에 도달했을 때 활성화
    approach_trigger_dist: float = 0.12

    # μ (grasp_trigger): λ AND cup_z > spawn_z + grasp_trigger_height → lift/goal 보상 활성화
    #   0.02m: 컵이 실제로 들렸을 때만 (테이블 마찰 오차 감안)
    grasp_trigger_height: float = 0.02

    # h2o 타겟 z offset: cup root frame → 실제 파지 중심
    # cup root가 바닥보다 0.015m 아래, 파지 중심은 root에서 0.056m 위
    object_grasp_z_offset: float = 0.056

    # -----------------------------------------------------------------------
    # DemoGrasp-style pregrasp reference (object-relative)
    # -----------------------------------------------------------------------
    # object local pose 기준 pregrasp 목표 (world +X,+Y,+Z 기준)
    # 기본값은 컵 뒤(-Y)에서 약간 위(+Z)로 접근하는 초기 레퍼런스
    pregrasp_offset_x: float = 0.0
    pregrasp_offset_y: float = -0.10
    pregrasp_offset_z: float = 0.06
    # 리셋 시 pregrasp 레퍼런스 랜덤화 범위
    pregrasp_noise_x: float = 0.02
    pregrasp_noise_y: float = 0.02
    pregrasp_noise_z: float = 0.01
    # pregrasp 근접 판정 및 보상
    pregrasp_activate_dist: float = 0.05
    pregrasp_reach_weight: float = 2.5
    pregrasp_reach_sharpness: float = 10.0
    # object-relative orientation reference (ZYX deg, world/object frame 기준)
    pregrasp_orient_offset_ez_deg: float = 90.0
    pregrasp_orient_offset_ey_deg: float = 0.0
    pregrasp_orient_offset_ex_deg: float = 90.0
    pregrasp_orient_noise_ez_deg: float = 8.0
    pregrasp_orient_noise_ey_deg: float = 8.0
    pregrasp_orient_noise_ex_deg: float = 8.0
    pregrasp_orient_activate_dist: float = 0.08
    pregrasp_orient_success_cos: float = 0.7
    pregrasp_orient_weight: float = 2.0
    pregrasp_orient_sharpness: float = 6.0

    # -----------------------------------------------------------------------
    # Tracking reference replay (DemoGrasp trackingReferenceFile style)
    # -----------------------------------------------------------------------
    # True면 policy action 대신 시계열 레퍼런스를 재생한다.
    use_reference_replay: bool = False
    # .pkl/.pt/.pth 파일 경로. 상대경로면 OPENARM_ROOT_DIR 기준으로 해석.
    tracking_reference_file: str = "assets/demograsp_references/normalized/grasp_ref_inspire_teosollo_pca5.pt"
    # grasp-only 모드에서 참조 시퀀스를 이 step까지만 재생(-1이면 마지막까지).
    tracking_reference_lift_timestep: int = -1
    # 리셋 시 레퍼런스 시작 pose 랜덤화(객체 local frame 기준)
    reference_pos_noise_x: float = 0.015
    reference_pos_noise_y: float = 0.015
    reference_pos_noise_z: float = 0.010
    reference_orient_noise_ez_deg: float = 6.0
    reference_orient_noise_ey_deg: float = 6.0
    reference_orient_noise_ex_deg: float = 6.0
    # 리셋 시 PCA 레퍼런스 바이어스(전 step 공통 오프셋)
    reference_pca_noise_scale: float = 0.05

    # -----------------------------------------------------------------------
    # Grasp-only stage (stage-3 split)
    # -----------------------------------------------------------------------
    grasp_only_mode: bool = True
    terminate_on_grasp_success: bool = True
    grasp_success_hold_steps: int = 8
    use_lift_success_for_final_success: bool = True
    grasp_success_palm_dist: float = 0.045
    grasp_success_hand_error: float = 0.12
    grasp_success_max_height_delta: float = 0.08
    grasp_stability_weight: float = 4.0
    # grasp-only 단계에서는 lift/goal 보상을 0으로 두는 것이 기본
    grasp_only_goal_reward_scale: float = 0.0
    grasp_only_lift_reward_scale: float = 0.0

    # -----------------------------------------------------------------------
    # Tip contact sensing (DemoGrasp-style minimal)
    # -----------------------------------------------------------------------
    use_tip_contact_gate: bool = True
    tip_object_contact_threshold: float = 0.02
    tip_table_contact_threshold: float = 0.02
    # Phase C2: force 신호가 유효하지 않을 때 fallback 분류 사용
    tip_contact_fallback_enabled: bool = True
    tip_force_valid_eps: float = 1e-6
    tip_object_fallback_dist: float = 0.055
    table_top_z: float = 0.25
    tip_table_fallback_margin: float = 0.010
    table_contact_penalty_weight: float = 0.5
    object_impact_penalty_weight: float = 0.15
    object_impact_force_threshold: float = 0.5
    use_self_collision_penalty: bool = False
    self_collision_penalty_weight: float = 0.05
    self_collision_force_threshold: float = 0.5
    right_tip_contact_links: tuple[str, ...] = (
        "rl_dg_1_4",
        "rl_dg_2_4",
        "rl_dg_3_4",
        "rl_dg_4_4",
        "rl_dg_5_4",
    )
    right_palm_contact_link: str = "rl_dg_palm"
    palm_object_contact_threshold: float = 0.02
    palm_table_contact_threshold: float = 0.02
    tip_contact_sensor_history_length: int = 1

    # -----------------------------------------------------------------------
    # 컵 기울기 종료 (displacement_penalty 제거: h2o 모순 유발)
    # -----------------------------------------------------------------------
    # cup_tipping: 컵 기울기가 max_tilt_deg 초과 → 에피소드 종료
    # 극단적 밀기(넘어뜨리기) 방지. 접근 중 살짝 밀리는 것은 불가피하므로 허용
    cup_tipping_max_deg: float = 60.0

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.55   # palm 시작 x≈0.46 → 컵을 palm 앞에 배치, +x+y 방향 접근
    object_spawn_y_center: float = -0.15  # palm workspace y∈[-0.50,-0.05] 중앙 근처; -Y 방향 접근
    object_spawn_z: float = 0.38        # visdex 물체 z_min=-0.076 기준 안전 높이
    object_spawn_xy_range: float = 0.08  # ±0.08m 균등 분포
    # 물체 활성 플래그
    enable_cup: bool = True
    enable_primitives: bool = False

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
            # 과도한 선할당을 줄여 DRAM/VRAM 압박 완화
            gpu_found_lost_aggregate_pairs_capacity=8 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=2 * 1024 * 1024,
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**22,
            gpu_collision_stack_size=2**22,
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
            activate_contact_sensors=True,
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
            activate_contact_sensors=True,
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
                # 오른팔: FK 탐색 결과 → palm ≈ [0.463, -0.311, 0.431]
                # 컵(y=-0.15)에서 0.16m 뒤(-y)에서 시작, +y 방향으로 직진 파지
                # j7=0.0: Fabrics default_config와 일치 (j7=1.571은 palm을 y=-0.01로 밀어냄)
                "openarm_right_joint1":  0.5,
                "openarm_right_joint2":  0.1,
                "openarm_right_joint3":  0.4,
                "openarm_right_joint4":  0.8,
                "openarm_right_joint5": -0.2,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                # 오른손: fully open (thumb opposition=0, 컵 소환 영역 관통 방지)
                "rj_dg_1_1": 0.0, "rj_dg_1_2":  0.0, "rj_dg_1_3": 0.0, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2":  0.0, "rj_dg_2_3": 0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2":  0.0, "rj_dg_3_3": 0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2":  0.0, "rj_dg_4_3": 0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2":  0.0, "rj_dg_5_3": 0.0, "rj_dg_5_4": 0.0,
                # 왼팔/왼쪽 그리퍼: 학습 제외, 고정 자세
                **LEFT_ARM_REST_JOINT_POS,
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

    # Right hand distal single-body contact sensor template
    # NOTE:
    #   rl_dg_*_tip prim 자체는 rigid-body API가 없어 ContactSensor를 직접 부착할 수 없다.
    #   따라서 rigid body인 rl_dg_*_4에 센서를 부착하고, 해당 바디 하위 tip collision(*_tip_c)을 포함해 측정한다.
    tip_single_body_contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_1_4",
        history_length=1,
        track_air_time=False,
    )

    # -----------------------------------------------------------------------
    # 컵 설정 (단일 컵 파지)
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, -0.15, 0.38],   # 초기 USD 위치 (실제 spawn은 _reset_idx에서 결정)
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "cup_bead/cup.usd"),
            activate_contact_sensors=True,
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
    # Primitive 설정 (초기 위치는 scene 밖, reset 시 랜덤 배정)
    # -----------------------------------------------------------------------
    primitive_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Primitive",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, -10.0],   # 초기에는 scene 밖 아래에 배치
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "primitives/USD/small_5_cyl/small_5_cyl.usd"),
            activate_contact_sensors=True,
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
    hand_body_names: list = HAND_BODY_NAMES_USD

    # -----------------------------------------------------------------------
    # Actuated joint names (오른팔 + 오른손, 27 DOF)
    # -----------------------------------------------------------------------
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES

    # 왼팔+왼쪽 그리퍼 joint names (고정, openarm_modular_dual.usd 기준)
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
