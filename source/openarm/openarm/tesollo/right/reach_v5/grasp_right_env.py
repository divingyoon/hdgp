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

"""환경 클래스: open-tesol_r_reach_v5

순수 접근(Pure Reach) 태스크를 위한 클린 DirectRLEnv 구현:
- Action (6D): 6D Palm Pose Target → Fabrics IK → Arm 7-DOF
- Hand (20-DOF): 접근 대기 자세 (HAND_APPROACH_POSE)로 고정
- Observation (37D): 대칭 기구학 관측 (Joint pos/vel, Palm/Cup 6D Pose, Rel Vector, Last Action)
- Reward: 가우시안 거리 커널 + 손바닥 법선(+X) 정렬 + 액션 스무딩
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from collections.abc import Sequence

import torch
import torch.nn.functional as F

# Fabrics 경로 설정 (hdgp/source/FABRICS/src 우선)
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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_ROBOT_DOF,
    NUM_PALM_ACTION,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    NUM_CRITIC_OBSERVATIONS,
    EPISODE_STEPS,
    REACH_SUCCESS_XY_THRESHOLD,
    REACH_SUCCESS_Z_THRESHOLD,
    REACH_ALIGNMENT_THRESHOLD_DEG,
    REACH_SUCCESS_HOLD_STEPS,
    ARM_START_POSE,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import (
    HAND_APPROACH_POSE,
    HAND_BODY_NAMES_USD,
    PREGRASP_OFFSET,
)
from .grasp_right_utils import scale, tensor_clamp, to_torch


class GraspRightEnv(DirectRLEnv):
    """open-tesol_r_reach_v5 직접 구동 RL 환경."""

    cfg: GraspRightEnvCfg

    def __init__(self, cfg: GraspRightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----------------------------------------------------------------
        # 1. 관절(DOF) 인덱스 매핑
        # ----------------------------------------------------------------
        self.actuated_dof_indices: list[int] = [
            self.robot.joint_names.index(name) for name in self.cfg.actuated_joint_names
        ]
        self.left_arm_dof_indices: list[int] = [
            self.robot.joint_names.index(name)
            for name in self.cfg.left_arm_joint_names
            if name in self.robot.joint_names
        ]
        self.arm_dof_indices = self.actuated_dof_indices[:NUM_ARM_DOF]    # 우팔 7개
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]  # 우손 20개

        # ----------------------------------------------------------------
        # 2. 바디(Link) 인덱스 매핑
        # ----------------------------------------------------------------
        _palm_name = "r_hl_palm"
        self.palm_body_index: int = (
            self.robot.data.body_names.index(_palm_name)
            if _palm_name in self.robot.data.body_names
            else -1
        )

        # ----------------------------------------------------------------
        # 3. 작업 공간 가두리 (Workspace Bounding Box)
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(PALM_POSE_MINS_FUNC(self.cfg.max_pose_angle), device=self.device)
        self.palm_maxs = to_torch(PALM_POSE_MAXS_FUNC(self.cfg.max_pose_angle), device=self.device)

        # ----------------------------------------------------------------
        # 4. 델타 액션 범위 (Delta Action Bounds)
        # ----------------------------------------------------------------
        _delta_rad = math.radians(self.cfg.palm_delta_rot_deg)
        _dx, _dy, _dz = self.cfg.palm_delta_xyz
        self.delta_mins = to_torch([-_dx, -_dy, -_dz, -_delta_rad, -_delta_rad, -_delta_rad], device=self.device)
        self.delta_maxs = to_torch([_dx, _dy, _dz, _delta_rad, _delta_rad, _delta_rad], device=self.device)

        # ----------------------------------------------------------------
        # 5. 로봇 및 손 고정 프리셋 포즈
        # ----------------------------------------------------------------
        self.hand_approach_pose = to_torch(HAND_APPROACH_POSE, device=self.device)
        self.arm_start_pose = to_torch(ARM_START_POSE, device=self.device)

        self.robot_start_joint_pos = torch.zeros(self.num_envs, NUM_ROBOT_DOF, device=self.device)
        self.robot_start_joint_pos[:, :NUM_ARM_DOF] = self.arm_start_pose.unsqueeze(0)
        self.robot_start_joint_pos[:, NUM_ARM_DOF:] = self.hand_approach_pose.unsqueeze(0)

        # 좌팔 고정 포즈
        self.left_arm_zero_pos = torch.zeros(self.num_envs, len(self.left_arm_dof_indices), device=self.device)
        for i, name in enumerate(self.cfg.left_arm_joint_names):
            if name in self.cfg.robot_cfg.init_state.joint_pos:
                self.left_arm_zero_pos[:, i] = self.cfg.robot_cfg.init_state.joint_pos[name]

        # ----------------------------------------------------------------
        # 6. 버퍼 할당
        # ----------------------------------------------------------------
        self.actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.pregrasp_palm_pose_buf = torch.zeros(self.num_envs, 6, device=self.device)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.reach_success_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Fabrics 제어점 버퍼
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)
        self.timestep = float(self.cfg.fabrics_dt)
        self.fabric_damping_gain = float(self.cfg.fabrics_damping_gain)

        # ----------------------------------------------------------------
        # 7. Fabrics IK 엔진 및 기구학 초기화
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()
        self._load_object_physical_tensors()
        self._build_home_pose()

    # ----------------------------------------------------------------------
    # 씬 구성 (Scene Setup)
    # ----------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=True)

        self.cup = RigidObject(self.cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup

        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ----------------------------------------------------------------------
    # 물체 물리 텐서 로딩
    # ----------------------------------------------------------------------
    def _load_object_physical_tensors(self) -> None:
        path = Path(self.cfg.object_bbox_path)
        if not path.is_file():
            raise FileNotFoundError(f"물체 bbox 파일 없음: {path}")
        table = json.loads(path.read_text(encoding="utf-8"))

        _REF = "cup_big_s100"
        ref_half_z = float(table[_REF][2])
        table_surface_z = float(self.cfg.object_spawn_z) - ref_half_z
        z_offset_ratio = float(self.cfg.pregrasp_offset_z) / ref_half_z

        self.object_idx = torch.tensor(
            [i % len(self.cfg.active_object_names) for i in range(self.num_envs)],
            dtype=torch.long,
            device=self.device,
        )
        half_extents = to_torch(
            [table[n] for n in self.cfg.active_object_names], device=self.device
        )  # (N_obj, 3)

        spawn_z_per_obj = table_surface_z + half_extents[:, 2]
        z_offset_per_obj = z_offset_ratio * half_extents[:, 2]

        self.object_spawn_z_buf = spawn_z_per_obj[self.object_idx]
        self.cup_grasp_z_offset_buf = z_offset_per_obj[self.object_idx]

    # ----------------------------------------------------------------------
    # Fabrics IK 플래너 초기화
    # ----------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        warp_cache_dir = self.device[-1]
        initialize_warp(warp_cache_dir)

        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.cfg.fabrics_max_objects_per_env,
            device=self.device,
            world_filename="open_tesollo_boxes_no_table",
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = float(self.cfg.fabrics_dt)

        # Main fabric (arm 제어용, graph_capturable=False)
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs,
            self.device,
            self.timestep,
            graph_capturable=False,
            use_hand_fabric=False,
            robot_dir_name="openarm_tesollo_bi_s",
            robot_name="openarm_tesollo_bi_s",
        )
        num_joints = self.open_tesollo_fabric.num_joints  # 27

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        # Fabric 상태 버퍼
        self.fabric_q = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabric input 버퍼
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self.fabric_damping_gain = self.cfg.fabrics_damping_gain * torch.ones(self.num_envs, 1, device=self.device)

    # ----------------------------------------------------------------------
    # 고정 홈 포즈 사전 계산
    # ----------------------------------------------------------------------
    def _build_home_pose(self) -> None:
        home_pose_deg = self.cfg.reset_home_palm_pose
        self.home_palm_pose = to_torch([
            home_pose_deg[0], home_pose_deg[1], home_pose_deg[2],
            math.radians(home_pose_deg[3]),
            math.radians(home_pose_deg[4]),
            math.radians(home_pose_deg[5]),
        ], device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # 초기 홈 관절각 IK 계산
        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets,
            self.home_palm_pose,
            "euler_zyx",
            self.robot_start_joint_pos,
            torch.zeros_like(self.robot_start_joint_pos),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        )
        _q = self.robot_start_joint_pos.clone()
        _qd = torch.zeros_like(_q)
        _qdd = torch.zeros_like(_q)
        for _ in range(60):
            _q, _qd, _qdd = self.open_tesollo_integrator.step(_q, _qd, _qdd, self.timestep)

        self.home_arm_joint_pos = _q[:, :NUM_ARM_DOF].clone()
        self.robot_start_joint_pos[:, :NUM_ARM_DOF] = self.home_arm_joint_pos
        self.fabric_q.copy_(self.robot_start_joint_pos)

    # ----------------------------------------------------------------------
    # 물리 스텝 사전 처리 (RL Action -> Fabrics IK -> Joint Target)
    # ----------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()

        # 1. 액션 스케일링 (6D Palm Pose Target)
        palm_action = actions[:, :NUM_PALM_ACTION]
        delta = scale(palm_action, self.delta_mins, self.delta_maxs)
        palm_pose = self.pregrasp_palm_pose_buf + delta

        # 2. 안전 작업 공간 클램프
        palm_pose = tensor_clamp(palm_pose, self.palm_mins.unsqueeze(0), self.palm_maxs.unsqueeze(0))
        self.palm_pose_targets.copy_(palm_pose)

        # 3. Fabrics IK 계산
        self.open_tesollo_fabric.set_features(
            self.hand_pca_targets,
            self.palm_pose_targets,
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.object_ids,
            self.object_indicator,
            self.fabric_damping_gain,
        )
        for _ in range(self.cfg.fabric_decimation):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.open_tesollo_integrator.step(
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_qdd.detach(),
                self.timestep,
            )

    # ----------------------------------------------------------------------
    # 관절 지령 인가
    # ----------------------------------------------------------------------
    def _apply_action(self) -> None:
        # 1. 우팔: Fabrics가 계산한 7-DoF 관절 목표각
        arm_target = self.fabric_q[:, :NUM_ARM_DOF]
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_dof_indices)
        self.robot.set_joint_velocity_target(torch.zeros_like(arm_target), joint_ids=self.arm_dof_indices)

        # 2. 우손: HAND_APPROACH_POSE 로 고정
        hand_target = self.hand_approach_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.robot.set_joint_position_target(hand_target, joint_ids=self.hand_dof_indices)
        self.robot.set_joint_velocity_target(torch.zeros_like(hand_target), joint_ids=self.hand_dof_indices)

        # 3. 유휴 좌팔: 대칭 미러 자세 고정
        self.robot.set_joint_position_target(self.left_arm_zero_pos, joint_ids=self.left_arm_dof_indices)

    # ----------------------------------------------------------------------
    # 기구학 중간값 계산 (Intermediate Kinematic Values)
    # ----------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # 1. 손바닥 위치 및 쿼터니언 (World & Local Frame)
        self.palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_index]
        self.palm_quat_w = self.robot.data.body_quat_w[:, self.palm_body_index]
        self.palm_pos = self.palm_pos_w - self.scene.env_origins

        # 2. 컵(타겟) 위치 및 쿼터니언 (World & Local Frame)
        self.object_pos_w = self.cup.data.root_pos_w
        self.object_quat_w = self.cup.data.root_quat_w
        self.object_pos = self.object_pos_w - self.scene.env_origins

        # 3. 컵 파지 중심점 (Grasp Center - Z 높이 보정 포함)
        self.grasp_center = self.object_pos.clone()
        self.grasp_center[:, 2] += self.cup_grasp_z_offset_buf

        # 4. [직교 분리] XY 수평 거리 및 Z 높이 오차
        palm_to_cup_xy = self.grasp_center[:, :2] - self.palm_pos[:, :2]
        self.palm_to_cup_dist_xy = palm_to_cup_xy.norm(dim=-1)
        self.palm_to_cup_dist_z = torch.abs(self.palm_pos[:, 2] - self.grasp_center[:, 2])

        # 5. [동적 상대 방향 벡터] 손바닥 -> 컵 수평 상대 벡터
        self.palm_to_cup_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_to_cup_dir[:, :2] = palm_to_cup_xy / self.palm_to_cup_dist_xy.unsqueeze(-1).clamp(min=1e-6)

        # 6. [2축 3D 공간 정렬]
        # (A) 손바닥 피부 정면 (+X 축) -> 컵 상대 방향 대면 정렬
        palm_normal_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        self.palm_normal_w = quat_apply(self.palm_quat_w, palm_normal_local)
        self.palm_alignment = torch.sum(self.palm_normal_w * self.palm_to_cup_dir, dim=-1).clamp(min=0.0)

        # (B) 손가락 뻗기 방향 (+Z 축) -> 바닥(-Z_world) 하향 정렬
        palm_down_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        palm_down_world = quat_apply(self.palm_quat_w, palm_down_local)
        self.palm_down_alignment = (-palm_down_world[:, 2]).clamp(min=0.0)

        # 7. [컵 외란 및 상태 추적]
        self.cup_xy_displacement = (self.object_pos[:, :2] - self.object_init_pos[:, :2]).norm(dim=-1)
        z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        cup_z_world = quat_apply(self.object_quat_w, z_local)
        self.cup_tilt_deg = torch.rad2deg(torch.acos(cup_z_world[:, 2].clamp(min=-1.0, max=1.0)))
        self.cup_lin_spd = self.cup.data.root_lin_vel_w.norm(dim=-1)
        self.cup_ang_spd = self.cup.data.root_ang_vel_w.norm(dim=-1)

    # ----------------------------------------------------------------------
    # 관측 텐서 반환 (Observation - 37D)
    # ----------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        # 관측 노이즈 주입 (Sim2Real Domain Randomization)
        σ_qp = self.cfg.obs_noise_joint_pos
        σ_qv = self.cfg.obs_noise_joint_vel
        σ_bp = self.cfg.obs_noise_body_pos
        σ_cp = self.cfg.obs_noise_cup_pos

        arm_pos_clean = self.robot.data.joint_pos[:, self.arm_dof_indices]
        arm_vel_clean = self.robot.data.joint_vel[:, self.arm_dof_indices]
        palm_pos_clean = self.palm_pos
        cup_pos_clean = self.object_pos

        arm_pos = arm_pos_clean + torch.randn_like(arm_pos_clean) * σ_qp
        arm_vel = arm_vel_clean + torch.randn_like(arm_vel_clean) * σ_qv
        palm_pos = palm_pos_clean + torch.randn_like(palm_pos_clean) * σ_bp
        cup_pos = cup_pos_clean + torch.randn_like(cup_pos_clean) * σ_cp

        # Actor Policy 관측 (37D - 노이즈 포함, 로봇 기준 Local 좌표계)
        policy_obs = torch.cat([
            arm_pos,                                   # (N, 7) 팔 관절 위치
            arm_vel * 0.1,                             # (N, 7) 팔 관절 속도 스케일링
            palm_pos,                                  # (N, 3) 손바닥 위치 (Local)
            self.palm_quat_w,                          # (N, 4) 손바닥 쿼터니언
            cup_pos,                                   # (N, 3) 컵 위치 (Local)
            self.object_quat_w,                        # (N, 4) 컵 쿼터니언
            cup_pos - palm_pos,                        # (N, 3) 상대 거리 벡터
            self.actions,                              # (N, 6) 직전 액션
        ], dim=-1)

        # Critic 관측 (37D - Clean 물리 진리값, 로봇 기준 Local 좌표계)
        critic_obs = torch.cat([
            arm_pos_clean,
            arm_vel_clean * 0.1,
            palm_pos_clean,
            self.palm_quat_w,
            cup_pos_clean,
            self.object_quat_w,
            cup_pos_clean - palm_pos_clean,
            self.actions,
        ], dim=-1)

        return {"policy": policy_obs, "critic": critic_obs}

    # ----------------------------------------------------------------------
    # 보상 계산 (Rewards - Reach v3 Decoupled Orthogonal Approach)
    # ----------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 1. [XY 수평 거리 보상] Standoff 8cm 목표 (선형 끌어당김 + Tanh 정밀 도달)
        dist_xy_error = torch.abs(self.palm_to_cup_dist_xy - self.cfg.standoff_target_dist)
        reward_xy = (
            - self.cfg.approach_xy_weight * dist_xy_error
            + self.cfg.approach_xy_fine_weight * (1.0 - torch.tanh(dist_xy_error / self.cfg.approach_xy_fine_std))
        )

        # 2. [Z 수직 높이 보상] 컵 허리 정중앙 높이 일치 (선형 끌어당김 + Tanh 정밀 일치)
        height_error = self.palm_to_cup_dist_z
        reward_z = (
            - self.cfg.approach_z_weight * height_error
            + self.cfg.approach_z_fine_weight * (1.0 - torch.tanh(height_error / self.cfg.approach_z_fine_std))
        )

        # 3. [2축 자세 정렬 보상] 손바닥 피부 정면 컵 대면 + 4손가락 하향
        reward_align = (
            self.cfg.approach_align_weight * self.palm_alignment
            + self.cfg.approach_down_align_weight * self.palm_down_alignment
        )

        # 4. [외란 감점] 컵 충돌/밀림/기울어짐 페널티
        disturbance_penalty = (
            self.cfg.cup_lin_vel_penalty_weight * self.cup_lin_spd
            + self.cfg.cup_ang_vel_penalty_weight * self.cup_ang_spd
            + self.cfg.approach_xy_penalty_weight * self.cup_xy_displacement
            + self.cfg.approach_tilt_penalty_weight * torch.relu(self.cup_tilt_deg - self.cfg.tilt_penalty_margin_deg)
        )

        # 5. [행동 정규화 페널티]
        r_action_smooth = torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_smooth_weight
        r_action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1) * self.cfg.action_rate_weight
        r_joint_vel = torch.sum((self.robot.data.joint_vel[:, self.arm_dof_indices] * 0.1) ** 2, dim=-1) * self.cfg.joint_vel_weight

        # 6. [성공 판정 및 연속 유지(Hold) 카운트]
        is_xy_close = dist_xy_error < REACH_SUCCESS_XY_THRESHOLD
        is_z_close = height_error < REACH_SUCCESS_Z_THRESHOLD
        is_aligned = self.palm_alignment > math.cos(math.radians(REACH_ALIGNMENT_THRESHOLD_DEG))
        is_undisturbed = (self.cup_xy_displacement < 0.015) & (self.cup_lin_spd < 0.1)
        success_now = is_xy_close & is_z_close & is_aligned & is_undisturbed

        self.reach_success_hold_buf = torch.where(
            success_now,
            self.reach_success_hold_buf + 1,
            torch.zeros_like(self.reach_success_hold_buf),
        )
        is_success = self.reach_success_hold_buf >= REACH_SUCCESS_HOLD_STEPS
        self.episode_success_buf.copy_(is_success)

        r_success = torch.where(is_success, self.cfg.reach_success_bonus, 0.0)

        # 총 보상 합산
        total_reward = (
            reward_xy
            + reward_z
            + reward_align
            - disturbance_penalty
            + r_action_smooth
            + r_action_rate
            + r_joint_vel
            + r_success
        )

        # W&B 실시간 상세 메트릭 기록
        self.extras["reward/xy"] = reward_xy.mean()
        self.extras["reward/z"] = reward_z.mean()
        self.extras["reward/align"] = reward_align.mean()
        self.extras["reward/disturbance"] = disturbance_penalty.mean()
        self.extras["task/xy_error_cm"] = (dist_xy_error * 100.0).mean()
        self.extras["task/z_error_cm"] = (height_error * 100.0).mean()
        self.extras["task/reach_success_rate"] = self.episode_success_buf.float().mean()

        return total_reward

    # ----------------------------------------------------------------------
    # 에피소드 종료 판정 (Terminations)
    # ----------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # 1. 타임아웃 (Max Episode Steps)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 2. 유효 작업공간 이탈 (Out-of-Bounds - 로봇 기준 Local Frame)
        out_x = (self.palm_pos[:, 0] < self.palm_mins[0]) | (self.palm_pos[:, 0] > self.palm_maxs[0])
        out_y = (self.palm_pos[:, 1] < self.palm_mins[1]) | (self.palm_pos[:, 1] > self.palm_maxs[1])
        out_z = (self.palm_pos[:, 2] < self.palm_mins[2]) | (self.palm_pos[:, 2] > self.palm_maxs[2])
        cup_fallen = self.object_pos[:, 2] < 0.20  # 컵이 테이블 밑으로 추락
        cup_knocked = self.cup_tilt_deg > 60.0    # 컵이 완전히 쓰러짐
        out_of_bounds = out_x | out_y | out_z | cup_fallen | cup_knocked

        # 3. 도달 성공 시 조기 완료
        died = out_of_bounds | self.episode_success_buf

        return died, time_out

    # ----------------------------------------------------------------------
    # 에피소드 리셋 (Reset Environment)
    # ----------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # 1. 로봇 관절 초기화 (Start Pose + 미세 노이즈)
        q_init = self.robot_start_joint_pos[env_ids].clone()
        noise = (torch.rand(n, NUM_ARM_DOF, device=self.device) - 0.5) * 2.0 * 0.02
        q_init[:, :NUM_ARM_DOF] += noise

        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.actuated_dof_indices] = q_init
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_zero_pos[0]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids_tensor)

        # 2. 컵 위치 무작위 스폰 (로봇 기준 Local 좌표계 -> World 좌표계 변환)
        obj_x = self.cfg.object_spawn_x_center + (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.object_spawn_x_range
        obj_y = self.cfg.object_spawn_y_center + (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.object_spawn_y_range
        obj_z = self.object_spawn_z_buf[env_ids]

        obj_pos_local = torch.stack([obj_x, obj_y, obj_z], dim=1)
        self.object_init_pos[env_ids] = obj_pos_local

        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]
        obj_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(n, -1)

        self.cup.write_root_pose_to_sim(torch.cat([obj_pos_world, obj_rot], dim=-1), env_ids=env_ids_tensor)
        self.cup.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids_tensor)

        # 3. Fabrics IK 상태 초기화 (고정 홈 포즈 기준)
        self.pregrasp_palm_pose_buf[env_ids] = self.home_palm_pose[env_ids]
        self.fabric_q[env_ids] = q_init
        self.fabric_qd[env_ids] = 0.0
        self.fabric_qdd[env_ids] = 0.0

        # 4. 버퍼 및 통계 리셋
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.reach_success_hold_buf[env_ids] = 0
        self.episode_success_buf[env_ids] = False

        self._compute_intermediate_values()
