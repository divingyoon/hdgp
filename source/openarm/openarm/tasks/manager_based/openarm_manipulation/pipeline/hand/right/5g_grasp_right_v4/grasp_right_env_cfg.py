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

"""환경 설정: 5g_grasp_right_v4

OpenArm(7 DOF) + Teosllo(20 DOF) 오른손 단일 컵 파지 태스크.

v4 핵심 변경 (v3 대비):
1. Reset: FABRICS로 palm을 cup 근처(~12cm 옆)로 이동 후 RL 시작
   - 이식: rl-icub reset_model() IK pregrasp 전략
   - 효과: RL이 approach 전체를 탐색할 필요 없음 → 난이도 대폭 감소
2. 보상: delta 기반 (iCub 스타일)
   - approach_delta × 100 (접촉 전만)
   - diff_num_contacts ±1 (정수)
   - lift_delta × 1000
   - goal_reward +1
3. curl_gate 제거: 항상 손가락 delta 제어
4. 접촉 감지: 거리 기반 primary (force 의존 안 함)
5. 게이트 단순화: already_touched_with_2_fingers 하나
6. 관측 간소화: point cloud feature 제거 (152D)
"""

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

_HDGP_ROOT = _os.path.normpath(_os.path.join(OPENARM_ROOT_DIR, "../../../../../../"))
_ASSETS_DIR = _os.path.join(_HDGP_ROOT, "assets")


@configclass
class GraspRightEnvCfg(DirectRLEnvCfg):
    """5g_grasp_right_v4 환경 설정."""

    # -----------------------------------------------------------------------
    # 시뮬레이션 파라미터
    # -----------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation: int = 2
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    use_cuda_graph: bool = False

    # -----------------------------------------------------------------------
    # 관측·액션 공간 (actor = critic = 동일 152D, separate: False)
    # -----------------------------------------------------------------------
    observation_space: int = NUM_OBSERVATIONS  # 152D
    action_space: int = NUM_ACTIONS            # 11D
    state_space: int = NUM_OBSERVATIONS        # 152D (actor와 동일)

    num_observations: int = NUM_OBSERVATIONS
    num_actions: int = NUM_ACTIONS
    num_states: int = NUM_OBSERVATIONS

    # -----------------------------------------------------------------------
    # Fabrics 파라미터
    # -----------------------------------------------------------------------
    use_hand_fabric: bool = False
    # 손가락 curl 증분 제어 (iCub max_delta_qpos 방식)
    max_delta_hand_q: float = 0.05  # 스텝당 최대 변화량 (rad)
    # Palm pose 워크스페이스 각도 한계 (deg)
    max_pose_angle: float = 45.0
    fabrics_max_objects_per_env: int = 6

    # -----------------------------------------------------------------------
    # Reset pregrasp (FABRICS IK, rl-icub reset_model() 대응)
    # -----------------------------------------------------------------------
    # reset 시 FABRICS로 N스텝 실행하여 palm을 cup 근처로 이동
    pregrasp_fabric_steps: int = 60    # 물리 스텝 수 (1/60s × 60 = 1s 시뮬)
    # NOTE: ARM_START_POSE 팔이 이미 컵 side 7cm 근처에 있으므로 60스텝으로 충분
    # 필요 시 증가 가능 (트레이드오프: 정확도 vs 리셋 속도)
    # pregrasp 목표: cup_pos + offset (side approach, -Y 방향)
    # [0.0, -0.12, 0.05] → 컵 왼쪽(오른손 관점) 12cm, 위로 5cm
    pregrasp_offset_x: float = 0.0
    pregrasp_offset_y: float = -0.20
    pregrasp_offset_z: float = 0.05
    # reset noise (환경 다양화)
    pregrasp_noise_x: float = 0.02
    pregrasp_noise_y: float = 0.02
    pregrasp_noise_z: float = 0.01

    # -----------------------------------------------------------------------
    # 보상 파라미터 (delta 기반, iCub 스타일)
    # -----------------------------------------------------------------------
    # 1. approach_delta: (prev_palm_dist - cur_palm_dist) × scale (접촉 전만)
    approach_reward_scale: float = 100.0

    # 2. diff_num_contacts: 접촉 수 증가 +1, 감소 -decrease_scale (정수 단위)
    contact_increase_reward: float = 1.0
    contact_decrease_penalty: float = 0.5  # 슬립 패널티 (iCub 원본=1.0, 탐색 유지 위해 절반)

    # 3. lift_delta: Δz × scale × (n_contacts / n_fingers) (접촉 수 비례)
    lift_reward_scale: float = 1000.0
    lift_drop_penalty_scale: float = 1000.0  # 컵 낙하 패널티

    # 4. goal_reward: cup_z > spawn_z + goal_height_threshold 달성 시 +1
    goal_reward: float = 1.0
    goal_height_threshold: float = 0.08     # 8cm 들기 성공 기준

    # -----------------------------------------------------------------------
    # 접촉 감지 (거리 기반 primary, force 의존 안 함)
    # -----------------------------------------------------------------------
    # fingertip-to-cup center 거리 threshold
    tip_contact_dist_threshold: float = 0.07   # 7cm (cup radius 4cm + margin 3cm)
    # ContactSensor는 보조 확인 용도로만 유지 (force_threshold는 크게 설정)
    tip_object_contact_threshold: float = 0.01
    table_top_z: float = 0.25
    cup_grasp_z_offset: float = 0.056  # cup root → 파지 중심 높이 오프셋

    # -----------------------------------------------------------------------
    # 이미 접촉 상태 (already_touched_with_2_fingers gate)
    # -----------------------------------------------------------------------
    min_fingers_for_approach_stop: int = 2  # approach reward를 끄는 접촉 수
    min_fingers_for_lift: int = 2           # lift reward 활성화 최소 접촉 수

    # -----------------------------------------------------------------------
    # 성공 / 종료 판정
    # -----------------------------------------------------------------------
    lift_success_height: float = 0.08   # spawn_z + 이 높이 달성 → 성공
    cup_tipping_max_deg: float = 60.0
    # 물체 영역 이탈 종료 범위
    obj_out_x_min: float = 0.05
    obj_out_x_max: float = 0.85
    obj_out_y_min: float = -0.60
    obj_out_y_max: float = 0.25
    obj_fallen_z: float = 0.18

    # -----------------------------------------------------------------------
    # 물체 spawn
    # -----------------------------------------------------------------------
    object_spawn_x_center: float = 0.40
    object_spawn_y_center: float = -0.15
    object_spawn_z: float = 0.38
    object_spawn_xy_range: float = 0.06   # ±6cm

    # -----------------------------------------------------------------------
    # 시뮬레이션 설정
    # -----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=4,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
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
    # 테이블 설정
    # -----------------------------------------------------------------------
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5725, 0.003, 0.235],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "scene_objects/table.usd"),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )

    # -----------------------------------------------------------------------
    # 로봇 설정 (openarm_tesollo_sensor.usd: fingertip contact sensor 내장)
    # -----------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSETS_DIR, "openarm_tesollo_sensor/openarm_tesollo_sensor.usd"),
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
                # 오른팔: 시작 자세 (palm ≈ [0.463, -0.311, 0.431])
                "openarm_right_joint1":  0.5,
                "openarm_right_joint2":  0.1,
                "openarm_right_joint3":  0.4,
                "openarm_right_joint4":  0.8,
                "openarm_right_joint5": -0.2,
                "openarm_right_joint6":  0.0,
                "openarm_right_joint7":  0.0,
                # 오른손: 엄지 사전 접힘 (-0.5), 나머지 열린 자세
                "rj_dg_1_1": 0.0, "rj_dg_1_2":  0.0, "rj_dg_1_3": 0.0, "rj_dg_1_4": 0.0,
                "rj_dg_2_1": 0.0, "rj_dg_2_2": 0.0, "rj_dg_2_3": 0.0, "rj_dg_2_4": 0.0,
                "rj_dg_3_1": 0.0, "rj_dg_3_2": 0.0, "rj_dg_3_3": 0.0, "rj_dg_3_4": 0.0,
                "rj_dg_4_1": 0.0, "rj_dg_4_2": 0.0, "rj_dg_4_3": 0.0, "rj_dg_4_4": 0.0,
                "rj_dg_5_1": 0.0, "rj_dg_5_2": 0.0, "rj_dg_5_3": 0.0, "rj_dg_5_4": 0.0,
                # 왼팔: 고정
                **LEFT_ARM_REST_JOINT_POS,
            },
        ),
        actuators={
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
            "tesollo_hand_abduction": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_1"],
                stiffness=1.9,
                damping=7.5e-4,
            ),
            "tesollo_hand_curl": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_2"],
                stiffness=0.84,
                damping=3.3e-4,
            ),
            "tesollo_hand_pip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_3"],
                stiffness=0.43,
                damping=1.7e-4,
            ),
            "tesollo_hand_dip": ImplicitActuatorCfg(
                joint_names_expr=["rj_dg_[1-5]_4"],
                stiffness=0.13,
                damping=5.1e-5,
            ),
            "openarm_left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=400.0,
                damping=80.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # -----------------------------------------------------------------------
    # Fingertip contact sensor (보조 - 거리 기반 primary 대비)
    # -----------------------------------------------------------------------
    right_tip_contact_links: tuple = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )
    right_palm_contact_link: str = "rl_dg_palm"

    # -----------------------------------------------------------------------
    # 컵 설정 (단일 컵)
    # -----------------------------------------------------------------------
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.38],
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
    # Hand / joint 이름 (preset에서)
    # -----------------------------------------------------------------------
    hand_body_names: list = HAND_BODY_NAMES_USD
    actuated_joint_names: list = RIGHT_ACTUATED_JOINT_NAMES
    left_arm_joint_names: list = LEFT_ARM_AND_GRIPPER_JOINT_NAMES
