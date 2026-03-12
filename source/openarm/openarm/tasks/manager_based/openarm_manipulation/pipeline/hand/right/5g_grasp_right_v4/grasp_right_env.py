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

"""환경 클래스: 5g_grasp_right_v4

v4 핵심 원칙 (rl-icub-dexterous-manipulation 이식 + v3 실패 교훈):

1. Reset pregrasp (rl-icub reset_model IK 대응):
   - FABRICS로 palm을 cup 옆(-Y 방향 12cm)으로 N스텝 이동
   - RL은 fine-adjustment + 손가락 제어만 담당
   - v3 실패 원인: RL이 approach 전체를 탐색해야 해서 너무 어려웠음

2. delta 기반 보상 (iCub 원본 방식):
   - approach_delta × 100 (접촉 전만)
   - diff_num_contacts ±1 (정수, 정체 시 보상 없음)
   - lift_delta × 1000 (실제로 들릴 때만 보상)
   - goal_reward +1

3. curl_gate 제거:
   - v3 실패 원인: curl_gate가 손가락 delta를 98% 차단
   - v4: 항상 손가락 delta 제어 허용

4. 접촉 감지: 거리 기반 primary
   - v3 실패 원인: ContactSensor force=0 (PhysX 설정 문제)
   - v4: fingertip-to-cup-center < 7cm를 primary 접촉 판정
   - ContactSensor는 보조로만 유지

5. 보상 게이트: 단순 already_touched_with_2_fingers 하나
   - v3 실패 원인: 3중 게이트(curl/approach/min_fingers) 동시 만족 불가
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from collections.abc import Sequence

import torch

# Fabrics 경로 설정 (hdgp/source/FABRICS/src)
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "source":
        _vendored = _parent / "FABRICS" / "src"
        if _vendored.exists():
            _v = str(_vendored)
            if _v not in sys.path:
                sys.path.insert(0, _v)
        break

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_FINGERTIPS,
    NUM_OBSERVATIONS,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    FABRIC_HAND_BODY_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    RIGHT_HAND_JOINT_NAMES,
    HAND_CURL_JOINT_NAMES,
    HAND_FIXED_JOINT_NAMES,
    HAND_FIXED_JOINT_VALUES,
    HAND_PIP_JOINT_NAMES,
    HAND_DIP_JOINT_NAMES,
    DISTAL_RATIO_PIP,
    DISTAL_RATIO_DIP,
    CURL_JOINT_LIMITS_MIN,
    CURL_JOINT_LIMITS_MAX,
    HAND_START_POSE,
    HAND_GRASP_POSE,
    OBJECT_GOAL_POS,
    PREGRASP_OFFSET,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 (v4).

    액션: 11D
      - [0:6] palm pose (x, y, z, ez, ey, ex), 정규화 [-1, 1]
      - [6:11] curl delta (5D), 정규화 [-1, 1] → Δq

    v4 핵심:
      - Reset: FABRICS pregrasp → RL이 fine-adjust + 손가락 제어
      - 보상: delta 기반 (iCub 스타일)
      - 접촉: 거리 기반 primary
      - 게이트: already_touched_with_2_fingers 하나
    """

    cfg: GraspRightEnvCfg

    def __init__(self, cfg: GraspRightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----------------------------------------------------------------
        # DOF index 설정
        # ----------------------------------------------------------------
        self.actuated_dof_indices: list[int] = []
        for name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(name))

        self.left_arm_dof_indices: list[int] = []
        for name in cfg.left_arm_joint_names:
            if name in self.robot.joint_names:
                self.left_arm_dof_indices.append(self.robot.joint_names.index(name))

        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]

        # ----------------------------------------------------------------
        # Palm pose 워크스페이스
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device)

        # ----------------------------------------------------------------
        # Hand poses
        # ----------------------------------------------------------------
        self.grasp_pose = to_torch(HAND_GRASP_POSE, device=self.device)  # (20,)
        self.hand_pca_mins = torch.zeros(5, device=self.device)  # unused in v4, 호환용
        self.hand_pca_maxs = torch.zeros(5, device=self.device)

        # ----------------------------------------------------------------
        # 로봇 시작 자세 (손가락 열린 상태)
        # ----------------------------------------------------------------
        arm_start = to_torch(ARM_START_POSE, device=self.device)
        hand_start = to_torch(HAND_START_POSE, device=self.device)
        robot_start = torch.cat([arm_start, hand_start], dim=0)  # (27,)
        self.robot_start_joint_pos = robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        # ----------------------------------------------------------------
        # 왼팔 고정 자세
        # ----------------------------------------------------------------
        left_vals = [
            LEFT_ARM_REST_JOINT_POS.get(self.robot.joint_names[idx], 0.0)
            for idx in self.left_arm_dof_indices
        ]
        self.left_arm_zero_pos = (
            to_torch(left_vals, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.left_arm_zero_vel = torch.zeros(
            self.num_envs, len(self.left_arm_dof_indices), device=self.device
        )

        # ----------------------------------------------------------------
        # 목표 위치
        # ----------------------------------------------------------------
        self.object_goal = to_torch(OBJECT_GOAL_POS, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Pregrasp offset (cup side approach, -Y 방향)
        self.pregrasp_offset = to_torch(
            [cfg.pregrasp_offset_x, cfg.pregrasp_offset_y, cfg.pregrasp_offset_z],
            device=self.device,
        )

        # ----------------------------------------------------------------
        # 중간값 버퍼
        # ----------------------------------------------------------------
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.fingertip_pos = torch.zeros(self.num_envs, NUM_FINGERTIPS, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, cfg.num_actions, device=self.device)

        # delta 보상 계산용: 이전 스텝 값 캐시
        self.prev_palm_dist = torch.full((self.num_envs,), 1.0, device=self.device)
        self.prev_cup_z = torch.full((self.num_envs,), cfg.object_spawn_z, device=self.device)
        self.prev_num_contacts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 접촉 상태
        # ----------------------------------------------------------------
        # per-finger 접촉 마스크 (거리 기반 primary)
        self.tip_contact_per_finger = torch.zeros(
            self.num_envs, NUM_FINGERTIPS, dtype=torch.bool, device=self.device
        )
        self.num_contacts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # already_touched_with_2_fingers: 에피소드 내 persistent flag
        # (iCub: 2손가락 이상 접촉 후 approach reward 차단)
        self.already_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 컵 기울기 종료용
        self._cup_tipping_cos = math.cos(math.radians(cfg.cup_tipping_max_deg))

        # ----------------------------------------------------------------
        # Fabrics 초기화
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"] = self.cup
        self.scene.rigid_objects["table"] = self.table

        # ContactSensor (보조 - 거리 기반이 primary)
        self._tip_sensors: list[ContactSensor] = []
        for link_name in self.cfg.right_tip_contact_links:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Cup"],
                history_length=1,
                track_air_time=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self._tip_sensors.append(sensor)
            self.scene.sensors[f"tip_sensor_{link_name}"] = sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=True)

    # ------------------------------------------------------------------
    # Geometric Fabrics 초기화
    # ------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        warp_cache_dir = self.device[-1]
        initialize_warp(warp_cache_dir)

        print("=== GraspRightEnv v4: Creating Fabrics world ===")
        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep,
            graph_capturable=False,
            use_hand_fabric=self.cfg.use_hand_fabric,
        )
        num_joints = self.open_tesollo_fabric.num_joints  # 27

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        # Fabrics 상태 버퍼 (27 DOF)
        self.fabric_q = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabrics input 버퍼
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # cspace attractor = GRASP_POSE (손가락이 자연히 파지 방향으로)
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.grasp_pose.unsqueeze(0).repeat(self.num_envs, 1)
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        # Fabrics cspace damping gain
        self.fabric_damping_gain = 10.0 * torch.ones(self.num_envs, 1, device=self.device)

        # ---- 직접 PD 손가락 제어 버퍼 ----
        self.arm_dof_indices = self.actuated_dof_indices[:7]
        self.hand_dof_indices_fab = self.actuated_dof_indices[7:]

        _hand_names = RIGHT_HAND_JOINT_NAMES
        self.hand_curl_indices_in_hand = torch.tensor(
            [_hand_names.index(n) for n in HAND_CURL_JOINT_NAMES],
            dtype=torch.long, device=self.device,
        )
        self.hand_fixed_indices_in_hand = torch.tensor(
            [_hand_names.index(n) for n in HAND_FIXED_JOINT_NAMES],
            dtype=torch.long, device=self.device,
        )
        self.hand_pip_indices_in_hand = torch.tensor(
            [_hand_names.index(n) for n in HAND_PIP_JOINT_NAMES],
            dtype=torch.long, device=self.device,
        )
        self.hand_dip_indices_in_hand = torch.tensor(
            [_hand_names.index(n) for n in HAND_DIP_JOINT_NAMES],
            dtype=torch.long, device=self.device,
        )

        self.distal_ratio_pip = torch.tensor(DISTAL_RATIO_PIP, device=self.device)
        self.distal_ratio_dip = torch.tensor(DISTAL_RATIO_DIP, device=self.device)

        self.curl_limits_min = torch.tensor(CURL_JOINT_LIMITS_MIN, device=self.device)
        self.curl_limits_max = torch.tensor(CURL_JOINT_LIMITS_MAX, device=self.device)

        self.hand_fixed_values = torch.tensor(
            HAND_FIXED_JOINT_VALUES, dtype=torch.float32, device=self.device,
        )

        # curl targets 초기값 (완전 열린 상태)
        _start_curl = [HAND_START_POSE[1], HAND_START_POSE[5], HAND_START_POSE[9],
                       HAND_START_POSE[13], HAND_START_POSE[18]]
        self.hand_curl_targets = torch.tensor(
            _start_curl, device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 5)

        # Hand FK taskmap
        robot_dir_name = "openarm_tesollo"
        robot_name = "openarm_tesollo"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path, FABRIC_HAND_BODY_NAMES, self.num_envs, self.device
        )

        print("=== GraspRightEnv v4: Fabrics initialized ===")

    # ------------------------------------------------------------------
    # 접촉 감지 (거리 기반 primary)
    # ------------------------------------------------------------------
    def _update_contact_state(self) -> None:
        """Update per-finger contact mask using distance-based primary detection.

        거리 기반 (primary): fingertip-to-cup-center < tip_contact_dist_threshold
        - ContactSensor force가 0이어도 항상 작동
        - iCub의 geom overlap 감지 방식과 동일한 역할
        """
        grasp_center = self.object_pos.clone()
        grasp_center[:, 2] += self.cfg.cup_grasp_z_offset  # cup root → 파지 중심

        tip_to_cup = (self.fingertip_pos - grasp_center.unsqueeze(1)).norm(dim=-1)  # (N, 5)
        dist_contact = tip_to_cup < self.cfg.tip_contact_dist_threshold  # (N, 5)

        self.tip_contact_per_finger.copy_(dist_contact)

        # 이전 스텝 저장 (delta 계산용)
        self.prev_num_contacts.copy_(self.num_contacts)
        self.num_contacts.copy_(dist_contact.sum(dim=-1))  # (N,)

        # already_touched_with_2_fingers 갱신 (한 번 True면 에피소드 내 유지)
        newly_touched = self.num_contacts >= self.cfg.min_fingers_for_approach_stop
        self.already_touched = self.already_touched | newly_touched

    # ------------------------------------------------------------------
    # Physics step (Fabrics 실행)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """액션 파싱 및 Fabrics 실행.

        actions[:, :6]: palm pose (정규화 [-1, 1])
        actions[:, 6:]: curl delta (정규화 [-1, 1] → Δq)
        """
        self.actions = actions.clone()

        palm_action = actions[:, :6]   # (N, 6)
        curl_action = actions[:, 6:]   # (N, 5)

        # Palm pose 스케일
        palm_pose = scale(palm_action, self.palm_mins, self.palm_maxs)  # (N, 6)
        self.palm_pose_targets.copy_(palm_pose)

        # Curl delta 증분 제어 (curl_gate 없음: 항상 허용)
        delta_q = curl_action * self.cfg.max_delta_hand_q   # (N, 5)
        new_curl = self.hand_curl_targets + delta_q
        new_curl = torch.max(new_curl, self.curl_limits_min.unsqueeze(0))
        new_curl = torch.min(new_curl, self.curl_limits_max.unsqueeze(0))
        self.hand_curl_targets.copy_(new_curl)

        # Fabrics 실행 (arm 제어)
        self.inputs = [
            self.hand_pca_targets,   # (N, 5), 0으로 고정
            self.palm_pose_targets,  # (N, 6)
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        ]
        self.open_tesollo_fabric.set_features(*self.inputs)
        for _ in range(self.cfg.fabric_decimation):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.open_tesollo_integrator.step(
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_qdd.detach(),
                self.timestep,
            )

    def _compute_hand_joint_targets(self) -> torch.Tensor:
        """5D curl → 20D hand joint targets (iCub distal tendon coupling)."""
        targets = torch.zeros(self.num_envs, 20, device=self.device)

        # 고정 관절
        targets[:, self.hand_fixed_indices_in_hand] = self.hand_fixed_values.unsqueeze(0)

        # Curl 관절
        targets[:, self.hand_curl_indices_in_hand] = self.hand_curl_targets

        # PIP 커플링
        targets[:, self.hand_pip_indices_in_hand[0]] = (
            self.hand_curl_targets[:, 0].abs() * self.distal_ratio_pip[0]
        )
        for i in range(1, 4):
            targets[:, self.hand_pip_indices_in_hand[i]] = (
                self.hand_curl_targets[:, i] * self.distal_ratio_pip[i]
            )
        targets[:, self.hand_pip_indices_in_hand[4]] = (
            self.hand_curl_targets[:, 4] * self.distal_ratio_pip[4]
        )

        # DIP 커플링
        targets[:, self.hand_dip_indices_in_hand[0]] = (
            self.hand_curl_targets[:, 0].abs() * self.distal_ratio_dip[0]
        )
        for i in range(1, 4):
            targets[:, self.hand_dip_indices_in_hand[i]] = (
                self.hand_curl_targets[:, i] * self.distal_ratio_dip[i]
            )

        return targets

    def _apply_action(self) -> None:
        # 오른팔 (7D): Fabrics 제어
        self.robot.set_joint_position_target(
            self.fabric_q[:, :7], joint_ids=self.arm_dof_indices
        )
        self.robot.set_joint_velocity_target(
            self.fabric_qd[:, :7], joint_ids=self.arm_dof_indices
        )

        # 오른손 (20D): 직접 PD 위치 제어
        hand_targets = self._compute_hand_joint_targets()
        self.robot.set_joint_position_target(
            hand_targets, joint_ids=self.hand_dof_indices_fab
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(hand_targets), joint_ids=self.hand_dof_indices_fab
        )

        # 왼팔: 고정 자세
        self.robot.write_joint_state_to_sim(
            self.left_arm_zero_pos,
            self.left_arm_zero_vel,
            joint_ids=self.left_arm_dof_indices,
        )

    # ------------------------------------------------------------------
    # Intermediate values
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # 물체 위치
        self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.cup.data.root_quat_w

        # Hand FK (7 bodies × 3D)
        # [0]=palm_center, [1]=palm_x, [2:7]=fingertip
        hand_pos_flat, _ = self.hand_points_taskmap(self.fabric_q, None)  # (N, 21)
        all_pos = hand_pos_flat.view(self.num_envs, 7, 3)

        # palm_center: USD body를 우선 사용 (더 정확)
        if self.cfg.right_palm_contact_link in self.robot.data.body_names:
            palm_body_id = self.robot.data.body_names.index(self.cfg.right_palm_contact_link)
            self.palm_center_pos = (
                self.robot.data.body_pos_w[:, palm_body_id] - self.scene.env_origins
            )
        else:
            self.palm_center_pos = all_pos[:, 0, :]

        self.fingertip_pos = all_pos[:, 2:, :]  # (N, 5, 3)

        # 접촉 감지 갱신
        self._update_contact_state()

        # palm_dist to cup grasp center
        grasp_center = self.object_pos.clone()
        grasp_center[:, 2] += self.cfg.cup_grasp_z_offset
        self._cur_palm_dist = (self.palm_center_pos - grasp_center).norm(dim=-1)  # (N,)

    # ------------------------------------------------------------------
    # Observations (actor = critic = 동일 152D, DEXTRAH Teacher 방식)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        touch_binary = self.tip_contact_per_finger.float()  # (N, 5)

        # 152D: actor = critic = 동일 obs
        # fabric_q/qd: Fabrics가 real에서도 실행되므로 privileged가 아님
        # cup_pos/cup_rot: Teacher = ground truth, Student = Foundation Pose (같은 차원 유지)
        obs = torch.cat([
            self.robot.data.joint_pos[:, self.actuated_dof_indices],   # 27D
            self.robot.data.joint_vel[:, self.actuated_dof_indices],   # 27D
            self.palm_center_pos,                                       # 3D  (FK)
            self.fingertip_pos.view(self.num_envs, -1),                 # 15D (FK)
            self.fabric_q,                                              # 27D (Fabrics 적분 상태)
            self.fabric_qd,                                             # 27D (Fabrics 속도 상태)
            self.object_pos,                                            # 3D  (Teacher: GT / Student: Foundation Pose)
            self.object_rot,                                            # 4D  (Teacher: GT / Student: Foundation Pose)
            self.object_goal,                                           # 3D  (고정값)
            touch_binary,                                               # 5D  (fingertip sensor)
            self.actions,                                               # 11D (last action)
        ], dim=-1)  # 152D

        if obs.shape[1] != NUM_OBSERVATIONS:
            raise RuntimeError(
                f"Obs dim mismatch: {obs.shape[1]} != {NUM_OBSERVATIONS}"
            )

        return {"policy": obs, "critic": obs}

    # ------------------------------------------------------------------
    # Rewards (delta 기반, iCub 스타일)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cur_palm_dist = self._cur_palm_dist  # (N,)
        cur_cup_z = self.object_pos[:, 2]    # (N,)
        n_contacts = self.num_contacts       # (N,)
        prev_contacts = self.prev_num_contacts

        # ---- 1. approach_delta: (prev - cur) × scale (접촉 전만) ----
        approach_delta = (self.prev_palm_dist - cur_palm_dist) * self.cfg.approach_reward_scale
        approach_reward = torch.where(self.already_touched, torch.zeros_like(approach_delta), approach_delta)

        # ---- 2. diff_num_contacts: ±1 per finger change ----
        contact_delta = (n_contacts - prev_contacts).float()  # (N,)
        delta_contact_reward = (
            self.cfg.contact_increase_reward * contact_delta.clamp(min=0.0)
            - self.cfg.contact_decrease_penalty * contact_delta.clamp(max=0.0).abs()
        )

        # ---- 3. lift_delta: Δz × scale × (contacts / 5) ----
        delta_z = (cur_cup_z - self.prev_cup_z).clamp(min=-0.1, max=0.1)  # (N,), clamped for safety
        contact_scale = (n_contacts.float() / float(NUM_FINGERTIPS)).clamp(0.0, 1.0)  # (N,)

        lift_condition = (n_contacts >= self.cfg.min_fingers_for_lift) & (delta_z > 0.0)
        lift_reward = torch.where(
            lift_condition,
            delta_z * self.cfg.lift_reward_scale * contact_scale,
            torch.zeros_like(delta_z),
        )
        # 낙하 패널티 (이미 접촉 후 컵이 아래로)
        drop_condition = self.already_touched & (delta_z < 0.0)
        drop_penalty = torch.where(
            drop_condition,
            delta_z.abs() * self.cfg.lift_drop_penalty_scale,
            torch.zeros_like(delta_z),
        )

        # ---- 4. goal_reward: +1 once per episode ----
        lifted = cur_cup_z > (self.object_init_pos[:, 2] + self.cfg.goal_height_threshold)
        goal_reward = torch.where(lifted, torch.full_like(cur_cup_z, self.cfg.goal_reward), torch.zeros_like(cur_cup_z))

        total = approach_reward + delta_contact_reward + lift_reward - drop_penalty + goal_reward

        # prev 값 업데이트 (다음 스텝을 위해)
        self.prev_palm_dist.copy_(cur_palm_dist)
        self.prev_cup_z.copy_(cur_cup_z)

        # 로깅
        self.extras["approach_reward"] = approach_reward.mean()
        self.extras["delta_contact_reward"] = delta_contact_reward.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["drop_penalty"] = drop_penalty.mean()
        self.extras["goal_reward"] = goal_reward.mean()
        self.extras["palm_dist"] = cur_palm_dist.mean()
        self.extras["num_contacts"] = n_contacts.float().mean()
        self.extras["already_touched"] = self.already_touched.float().mean()
        self.extras["object_z"] = cur_cup_z.mean()
        self.extras["contact_delta"] = contact_delta.mean()
        self.extras["lift_success"] = lifted.float().mean()

        return total

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # 컵 영역 이탈
        out_x = (
            (self.object_pos[:, 0] < self.cfg.obj_out_x_min)
            | (self.object_pos[:, 0] > self.cfg.obj_out_x_max)
        )
        out_y = (
            (self.object_pos[:, 1] < self.cfg.obj_out_y_min)
            | (self.object_pos[:, 1] > self.cfg.obj_out_y_max)
        )
        fallen = self.object_pos[:, 2] < self.cfg.obj_fallen_z

        # 컵 기울기 초과
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)
        tipped = cup_z_world[:, 2] < self._cup_tipping_cos

        # 성공
        lifted = self.object_pos[:, 2] > (
            self.object_init_pos[:, 2] + self.cfg.lift_success_height
        )
        grasp_success = lifted & self.already_touched

        terminated = out_x | out_y | fallen | tipped | grasp_success
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["grasp_success"] = grasp_success.float().mean()
        self.extras["final_success"] = grasp_success.float().mean()

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset (FABRICS pregrasp 핵심 구현)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset with FABRICS pregrasp.

        rl-icub reset_model() IK pregrasp 전략 대응:
        1. 컵 spawn
        2. palm target = cup_pos + pregrasp_offset (side approach)
        3. FABRICS N스텝 실행 → palm이 pregrasp 위치로 이동
        4. RL 시작
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # ---- 1. 로봇 관절 상태 리셋 (start pose) ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        for k, idx in enumerate(self.actuated_dof_indices):
            full_pos[:, idx] = self.robot_start_joint_pos[0, k]
        for k, idx in enumerate(self.left_arm_dof_indices):
            full_pos[:, idx] = self.left_arm_zero_pos[0, k]

        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # ---- 2. Fabrics 상태 리셋 ----
        self.fabric_q[env_ids] = self.robot_start_joint_pos[env_ids]
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # ---- 3. 컵 목표 위치 계산 (아직 spawn 안 함) ----
        # 컵 spawn은 Fabrics pregrasp + 관절 쓰기 이후에 수행.
        # 이 순서 덕분에 컵이 생성되는 순간 팔이 이미 pregrasp 위치에 있어
        # 엄지/손가락이 컵을 관통하지 않는다.
        obj_x = self.cfg.object_spawn_x_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_y = self.cfg.object_spawn_y_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_pos_local = torch.stack(
            [obj_x, obj_y, torch.full((n,), self.cfg.object_spawn_z, device=self.device)], dim=1
        )

        self.object_init_pos[env_ids] = obj_pos_local

        # ---- 4. FABRICS pregrasp: cup 옆(-Y)으로 palm 이동 ----
        # pregrasp 목표 = cup_pos + offset + noise
        noise = torch.stack([
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
            (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
        ], dim=1)
        pregrasp_pos = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + noise  # (n, 3)

        # palm pose target: xyz + 기본 orientation (ez=90°, ey=90°, ex=90° → side approach)
        # PALM_POSE center: ez=0°, ey=+90°, ex=0° (memory에서)
        pregrasp_palm_pose = torch.zeros(n, 6, device=self.device)
        pregrasp_palm_pose[:, :3] = pregrasp_pos
        pregrasp_palm_pose[:, 3] = math.radians(90.0)   # ez = 90°
        pregrasp_palm_pose[:, 4] = math.radians(90.0)   # ey = 90° (손가락이 +X)
        pregrasp_palm_pose[:, 5] = math.radians(90.0)   # ex = 90°

        # palm_mins/maxs 클램프
        pregrasp_palm_pose = torch.max(
            torch.min(pregrasp_palm_pose, self.palm_maxs.unsqueeze(0)),
            self.palm_mins.unsqueeze(0),
        )

        # 해당 env에 palm target 적용하고 Fabrics N스텝 실행
        self.palm_pose_targets[env_ids] = pregrasp_palm_pose

        # N steps Fabrics 적분 (arm만, hand는 cspace attractor로 열린 자세 유지)
        hand_pca_zero = torch.zeros(n, 5, device=self.device)
        damping_gain_sub = self.fabric_damping_gain[env_ids]

        # 부분 Fabrics 실행을 위해 전체 N 배치 중 env_ids만 업데이트
        fabric_q_full = self.fabric_q.clone()
        fabric_qd_full = self.fabric_qd.clone()
        fabric_qdd_full = self.fabric_qdd.clone()

        # 전체 배치 inputs 구성 (env_ids 외의 envs는 현재 상태 그대로)
        pregrasp_palm_all = self.palm_pose_targets.clone()
        pregrasp_palm_all[env_ids] = pregrasp_palm_pose

        inputs_pregrasp = [
            self.hand_pca_targets,    # (N, 5), 0
            pregrasp_palm_all,        # (N, 6)
            "euler_zyx",
            fabric_q_full.detach(),
            fabric_qd_full.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        ]
        self.open_tesollo_fabric.set_features(*inputs_pregrasp)

        for _ in range(self.cfg.pregrasp_fabric_steps):
            fabric_q_full, fabric_qd_full, fabric_qdd_full = self.open_tesollo_integrator.step(
                fabric_q_full.detach(),
                fabric_qd_full.detach(),
                fabric_qdd_full.detach(),
                self.timestep,
            )

        # env_ids만 Fabrics 상태 업데이트
        self.fabric_q[env_ids] = fabric_q_full[env_ids]
        self.fabric_qd[env_ids] = fabric_qd_full[env_ids]
        self.fabric_qdd[env_ids] = fabric_qdd_full[env_ids]

        # ---- 5. 로봇 관절을 Fabrics 결과로 업데이트 ----
        pregrasp_full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        pregrasp_full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        # 오른팔 (7D): Fabrics 결과
        for k, idx in enumerate(self.arm_dof_indices):
            pregrasp_full_pos[:, idx] = self.fabric_q[env_ids, k]
        # 오른손 (20D): 열린 자세 (컵 위치 관통 방지)
        for k_hand, idx in enumerate(self.hand_dof_indices_fab):
            pregrasp_full_pos[:, idx] = HAND_START_POSE[k_hand]
        # 왼팔: 고정
        for k, idx in enumerate(self.left_arm_dof_indices):
            pregrasp_full_pos[:, idx] = self.left_arm_zero_pos[0, k]

        self.robot.write_joint_state_to_sim(pregrasp_full_pos, pregrasp_full_vel, env_ids=env_ids)

        # ---- 6. 컵 spawn (팔이 pregrasp 위치에 있은 후 생성) ----
        # 이 시점에서 palm은 이미 cup에서 ~20cm 떨어진 pregrasp 위치에 있음.
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0
        zero_vel = torch.zeros(n, 6, device=self.device)
        cup_root_state = torch.cat([obj_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # ---- 8. 손가락 curl targets 리셋 (완전 열린 상태) ----
        _start_curl = [HAND_START_POSE[1], HAND_START_POSE[5], HAND_START_POSE[9],
                       HAND_START_POSE[13], HAND_START_POSE[18]]
        self.hand_curl_targets[env_ids] = to_torch(_start_curl, device=self.device).unsqueeze(0).repeat(n, 1)

        # ---- 9. delta 보상 버퍼 리셋 ----
        grasp_center = obj_pos_local.clone()
        grasp_center[:, 2] += self.cfg.cup_grasp_z_offset
        palm_pregrasp_pos = self.fabric_q[env_ids, :3]  # Fabrics 결과 palm 위치 근사
        # 실제 palm_center는 FK 계산 전이므로 pregrasp_pos로 근사
        init_palm_dist = (pregrasp_pos - grasp_center).norm(dim=-1)
        self.prev_palm_dist[env_ids] = init_palm_dist
        self.prev_cup_z[env_ids] = self.cfg.object_spawn_z
        self.prev_num_contacts[env_ids] = 0

        # ---- 10. 접촉 상태 리셋 ----
        self.tip_contact_per_finger[env_ids] = False
        self.num_contacts[env_ids] = 0
        self.already_touched[env_ids] = False

