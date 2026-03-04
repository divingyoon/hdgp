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

"""환경 클래스: 5g_grasp_right_v1

컨트롤 방식: Geometric Fabrics (DEXTRAH 연계)
  - palm pose 6D → Fabrics → arm 7 DOF joint target
  - finger 1D → open/grasp 선형 보간 → hand 20 DOF joint target

물체: cup / mug 중 에피소드마다 랜덤 선택 (각 50%)

리워드: KUKA_ALLEGRO 방식 4항목
  1. hand_to_object: 손 전체(palm+5 tips)가 물체에 고르게 접근
  2. object_to_goal: 물체 → 목표 위치
  3. lift: 물체 수직 들기
  4. finger_curl_reg: 파지 자세 유지 패널티
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Fabrics imports (DEXTRAH 연계)
from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

from .grasp_right_env_cfg import GraspRightEnvCfg
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    HAND_OPEN_POSE,
    HAND_GRASP_POSE,
    ARM_START_POSE,
    OBJECT_GOAL_POS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경 (cup + mug 랜덤 선택).

    액션: 7D
      - [0:6] palm pose (x, y, z, ez, ey, ex), 정규화 [-1, 1]
      - [6]   finger (열림=-1, 닫힘=+1)

    Fabrics: arm(7 DOF)를 palm pose로 제어, hand(20 DOF)는 직접 보간.
    에피소드마다 cup/mug 중 하나를 랜덤 배정, 나머지는 scene 밖으로 이동.
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

        # 오른손 관절 indices (20 DOF): actuated_dof_indices 중 arm 이후
        self.hand_dof_indices = self.actuated_dof_indices[NUM_ARM_DOF:]

        # ----------------------------------------------------------------
        # Palm pose 워크스페이스
        # ----------------------------------------------------------------
        self.palm_mins = to_torch(
            PALM_POSE_MINS_FUNC(cfg.max_pose_angle), device=self.device
        )
        self.palm_maxs = to_torch(
            PALM_POSE_MAXS_FUNC(cfg.max_pose_angle), device=self.device
        )

        # ----------------------------------------------------------------
        # Hand poses (open / grasp)
        # ----------------------------------------------------------------
        self.open_pose = to_torch(HAND_OPEN_POSE, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)   # (N, 20)
        self.grasp_pose = to_torch(HAND_GRASP_POSE, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 20)

        # finger_curl_reg에서 target으로 사용할 grasp pose (1D 버전, for reward)
        self.grasp_pose_target = to_torch(HAND_GRASP_POSE, device=self.device)  # (20,)

        # ----------------------------------------------------------------
        # 로봇 시작 자세
        # ----------------------------------------------------------------
        arm_start = to_torch(ARM_START_POSE, device=self.device)  # (7,)
        hand_start = to_torch(HAND_GRASP_POSE, device=self.device)  # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)    # (27,)
        self.robot_start_joint_pos = robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        # ----------------------------------------------------------------
        # 왼팔 고정 자세
        # ----------------------------------------------------------------
        left_rest_dict = {
            "openarm_left_joint1": -0.5,
            "openarm_left_joint2": -0.5,
            "openarm_left_joint3":  0.6,
            "openarm_left_joint4":  0.7,
            "openarm_left_joint5":  0.0,
            "openarm_left_joint6":  0.0,
            "openarm_left_joint7": -1.0,
        }
        left_vals = [
            left_rest_dict.get(self.robot.joint_names[idx], 0.0)
            for idx in self.left_arm_dof_indices
        ]
        self.left_arm_zero_pos = (
            to_torch(left_vals, device=self.device)
            .unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.left_arm_zero_vel = torch.zeros(self.num_envs, len(self.left_arm_dof_indices), device=self.device)

        # ----------------------------------------------------------------
        # 목표 위치
        # ----------------------------------------------------------------
        self.object_goal = to_torch(OBJECT_GOAL_POS, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # ----------------------------------------------------------------
        # 중간값 버퍼
        # ----------------------------------------------------------------
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.hand_pos = torch.zeros(self.num_envs, 6, 3, device=self.device)  # 6 bodies × 3D
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # ----------------------------------------------------------------
        # 물체 유형 추적: 0=cup, 1=mug (에피소드마다 랜덤)
        # ----------------------------------------------------------------
        self.active_object_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # Fabrics 초기화 (arm 제어용)
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

        # ----------------------------------------------------------------
        # Hand FK taskmap (Fabrics URDF 기준, 6 bodies)
        # palm_center: palm_link에서 local offset = 실제 손바닥 중심
        # ----------------------------------------------------------------
        robot_dir_name = "openarm_tesollo"
        robot_name = "openarm_tesollo"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        fabric_hand_body_names = [
            "palm_center",
            "tesollo_right_rl_dg_1_4",
            "tesollo_right_rl_dg_2_4",
            "tesollo_right_rl_dg_3_4",
            "tesollo_right_rl_dg_4_4",
            "tesollo_right_rl_dg_5_4",
        ]
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path, fabric_hand_body_names, self.num_envs, self.device
        )

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup = RigidObject(self.cfg.cup_cfg)
        self.mug = RigidObject(self.cfg.mug_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"] = self.cup
        self.scene.rigid_objects["mug"] = self.mug
        self.scene.rigid_objects["table"] = self.table

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

    # ------------------------------------------------------------------
    # Geometric Fabrics 초기화 (DEXTRAH _setup_geometric_fabrics 기반)
    # ------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        warp_cache_dir = self.device[-1]  # GPU index
        initialize_warp(warp_cache_dir)

        print("=== GraspRightEnv: Creating Fabrics world ===")
        world_filename = "open_tesollo_boxes"
        max_objects_per_env = 20
        self.world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=max_objects_per_env,
            device=self.device,
            world_filename=world_filename,
        )
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        self.timestep = self.cfg.fabrics_dt

        # OpenArm+Teosllo 27 DOF Fabric
        self.open_tesollo_fabric = OpenArmTeoslloPoseFabric(
            self.num_envs, self.device, self.timestep, graph_capturable=True
        )
        num_joints = self.open_tesollo_fabric.num_joints  # 27

        self.open_tesollo_integrator = DisplacementIntegrator(self.open_tesollo_fabric)

        # Fabric 상태 버퍼 (27 DOF)
        self.fabric_q = self.robot_start_joint_pos.clone().contiguous()
        self.fabric_qd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Fabric input 버퍼
        # hand_pca_targets: 5D PCA (0으로 고정 — hand는 직접 보간으로 제어)
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)
        # palm_pose_targets: 6D palm pose (ez, ey, ex 순서 euler_zyx)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # Fabric cspace damping gain (고정값, ADR 없음)
        self.fabric_damping_gain = 10.0 * torch.ones(self.num_envs, 1, device=self.device)

        # 현재 hand interpolation 결과 버퍼
        self.current_hand_q = self.grasp_pose.clone()

        # CUDA graph capture
        if self.cfg.use_cuda_graph:
            self.inputs = [
                self.hand_pca_targets,
                self.palm_pose_targets,
                "euler_zyx",
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.object_ids,
                self.object_indicator,
                self.fabric_damping_gain,
            ]
            (
                self.g,
                self.fabric_q_new,
                self.fabric_qd_new,
                self.fabric_qdd_new,
            ) = capture_fabric(
                self.open_tesollo_fabric,
                self.fabric_q,
                self.fabric_qd,
                self.fabric_qdd,
                self.timestep,
                self.open_tesollo_integrator,
                self.inputs,
                self.device,
            )

        print("=== GraspRightEnv: Fabrics initialized ===")

    # ------------------------------------------------------------------
    # Physics step (Fabrics 실행)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        # ---- 액션 파싱 ----
        palm_action = actions[:, :6]    # (N, 6), 정규화 [-1, 1]
        finger_action = actions[:, 6]   # (N,), 정규화 [-1, 1]

        # Palm pose 스케일 (정규화 → 실제 workspace)
        palm_pose = scale(palm_action, self.palm_mins, self.palm_maxs)  # (N, 6)
        self.palm_pose_targets.copy_(palm_pose)

        # PCA targets = 0 (hand는 직접 보간 제어)
        self.hand_pca_targets.zero_()

        # ---- Hand 보간 ----
        # t ∈ [0, 1]: 0=open, 1=grasp
        t = ((finger_action + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(1)  # (N, 1)
        hand_q = (1.0 - t) * self.open_pose + t * self.grasp_pose        # (N, 20)
        self.current_hand_q.copy_(hand_q)

        # ---- Fabrics 실행 (arm 제어) ----
        if not self.cfg.use_cuda_graph:
            self.inputs = [
                self.hand_pca_targets,
                self.palm_pose_targets,
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
        else:
            for _ in range(self.cfg.fabric_decimation):
                self.g.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)

        # ---- Hand joints를 보간값으로 덮어씀 ----
        # Fabrics arm 출력만 사용하고, hand는 직접 보간값으로 설정
        self.fabric_q[:, NUM_ARM_DOF:] = self.current_hand_q.detach()
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _apply_action(self) -> None:
        # 오른팔+오른손: Fabrics arm (7D) + 보간 hand (20D)
        self.robot.set_joint_position_target(
            self.fabric_q, joint_ids=self.actuated_dof_indices
        )
        self.robot.set_joint_velocity_target(
            self.fabric_qd, joint_ids=self.actuated_dof_indices
        )

        # 왼팔+왼손: 매 스텝 고정 자세로 강제
        self.robot.write_joint_state_to_sim(
            self.left_arm_zero_pos,
            self.left_arm_zero_vel,
            joint_ids=self.left_arm_dof_indices,
        )

    # ------------------------------------------------------------------
    # Intermediate values (공통 계산)
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        # 물체 위치: active object에 따라 cup/mug 선택, world → env local
        cup_pos = self.cup.data.root_pos_w - self.scene.env_origins   # (N, 3)
        mug_pos = self.mug.data.root_pos_w - self.scene.env_origins   # (N, 3)
        cup_rot = self.cup.data.root_quat_w                           # (N, 4)
        mug_rot = self.mug.data.root_quat_w                           # (N, 4)

        use_mug = (self.active_object_type == 1).unsqueeze(1)         # (N, 1)
        self.object_pos = torch.where(use_mug.expand(-1, 3), mug_pos, cup_pos)
        self.object_rot = torch.where(use_mug.expand(-1, 4), mug_rot, cup_rot)

        # 손 위치: Fabrics FK (6 bodies × 3D = 18D)
        hand_pos_flat, _ = self.hand_points_taskmap(self.fabric_q, None)  # (N, 18)
        self.hand_pos = hand_pos_flat.view(self.num_envs, 6, 3)  # (N, 6, 3)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 주의: _get_dones에서 _compute_intermediate_values 호출됨
        # 추가 호출 시 중복이지만 안전을 위해 유지

        obs = torch.cat([
            # 로봇 관절 상태 (오른팔+오른손, 27 DOF)
            self.robot.data.joint_pos[:, self.actuated_dof_indices],   # (N, 27)
            self.robot.data.joint_vel[:, self.actuated_dof_indices],   # (N, 27)
            # Hand FK 위치 (6 bodies × 3D)
            self.hand_pos.view(self.num_envs, -1),                      # (N, 18)
            # 물체 상태
            self.object_pos,                                            # (N, 3)
            self.object_rot,                                            # (N, 4)
            # 목표 위치
            self.object_goal,                                           # (N, 3)
            # 마지막 액션
            self.actions,                                               # (N, 7)
            # Fabrics 상태 (arm 추론용)
            self.fabric_q,                                              # (N, 27)
            self.fabric_qd,                                             # (N, 27)
        ], dim=-1)  # (N, 143)

        return {"policy": obs, "critic": obs}

    # ------------------------------------------------------------------
    # Rewards (KUKA_ALLEGRO 방식)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- 1. hand_to_object ----
        # palm + 5 fingertips와 물체 간의 최대 거리 기반
        # 모든 손가락이 고르게 접근하도록 max 사용 (KUKA_ALLEGRO 방식)
        obj_expanded = self.object_pos.unsqueeze(1).expand(-1, 6, -1)  # (N, 6, 3)
        dist_per_body = (self.hand_pos - obj_expanded).norm(dim=-1)    # (N, 6)
        max_dist = dist_per_body.max(dim=-1).values                    # (N,)

        hand_to_obj_reward = self.cfg.hand_to_object_weight * torch.exp(
            -self.cfg.hand_to_object_sharpness * max_dist
        )

        # ---- 2. object_to_goal ----
        goal_dist = (self.object_pos - self.object_goal).norm(dim=-1)  # (N,)
        obj_to_goal_reward = self.cfg.object_to_goal_weight * torch.exp(
            -self.cfg.object_to_goal_sharpness * goal_dist
        )

        # ---- 3. lift ----
        # 물체 z와 목표 z의 차이 기반 (물체를 들어야 reward)
        lift_error = (self.object_pos[:, 2] - self.object_goal[:, 2]).abs()  # (N,)
        lift_reward = self.cfg.lift_weight * torch.exp(
            -self.cfg.lift_sharpness * lift_error
        )

        # ---- 4. finger_curl_reg ----
        # 파지 자세(grasp_pose)에서 벗어날수록 패널티 (닫힘 유도)
        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]          # (N, 20)
        curl_error = (hand_q - self.grasp_pose_target.unsqueeze(0)).pow(2).sum(dim=-1)  # (N,)
        curl_reg = self.cfg.finger_curl_weight * curl_error

        # ---- 합산 ----
        total = hand_to_obj_reward + obj_to_goal_reward + lift_reward + curl_reg

        # 로깅
        self.extras["hand_to_object_reward"] = hand_to_obj_reward.mean()
        self.extras["object_to_goal_reward"] = obj_to_goal_reward.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["finger_curl_reg"] = curl_reg.mean()
        self.extras["max_hand_dist"] = max_dist.mean()
        self.extras["goal_dist"] = goal_dist.mean()
        self.extras["object_z"] = self.object_pos[:, 2].mean()
        self.extras["mug_ratio"] = (self.active_object_type == 1).float().mean()

        return total

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # 물체가 작업 영역 밖으로 나갔을 때 종료
        out_x = (self.object_pos[:, 0] < 0.05) | (self.object_pos[:, 0] > 0.85)
        out_y = (self.object_pos[:, 1] < -0.55) | (self.object_pos[:, 1] > 0.25)
        fallen = self.object_pos[:, 2] < 0.18  # 테이블 위 최소 높이

        terminated = out_x | out_y | fallen
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # ---- 로봇 관절 상태 리셋 ----
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)

        # 오른팔+오른손: start pose
        for k, idx in enumerate(self.actuated_dof_indices):
            full_pos[:, idx] = self.robot_start_joint_pos[0, k]

        # 왼팔+왼손: 고정 자세
        for k, idx in enumerate(self.left_arm_dof_indices):
            full_pos[:, idx] = self.left_arm_zero_pos[0, k]

        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # ---- Fabrics 상태 리셋 ----
        self.fabric_q[env_ids] = self.robot_start_joint_pos[env_ids]
        self.fabric_qd[env_ids].zero_()
        self.fabric_qdd[env_ids].zero_()

        # ---- 물체 유형 랜덤 배정 (0=cup, 1=mug) ----
        self.active_object_type[env_ids] = torch.randint(
            0, 2, (n,), device=self.device, dtype=torch.long
        )
        is_cup = (self.active_object_type[env_ids] == 0)  # (n,) bool

        # ---- 활성 물체 spawn 위치 (랜덤 ±xy 오프셋) ----
        obj_x = self.cfg.object_spawn_x_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_y = self.cfg.object_spawn_y_center + (
            torch.rand(n, device=self.device) - 0.5
        ) * 2.0 * self.cfg.object_spawn_xy_range
        obj_pos_local = torch.stack([obj_x, obj_y,
                                     torch.full((n,), self.cfg.object_spawn_z, device=self.device)], dim=1)
        obj_pos_world = obj_pos_local + self.scene.env_origins[env_ids]  # (n, 3)

        # ---- inactive 물체는 scene 밖 아래로 이동 ----
        offscene_pos = self.scene.env_origins[env_ids].clone()
        offscene_pos[:, 2] -= 10.0  # env local z=-10m (테이블 아래)

        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0   # w=1 (identity quaternion)
        zero_vel = torch.zeros(n, 6, device=self.device)

        # cup: active이면 spawn 위치, inactive이면 scene 밖
        cup_pos_world = torch.where(is_cup.unsqueeze(1), obj_pos_world, offscene_pos)
        cup_root_state = torch.cat([cup_pos_world, upright_rot, zero_vel], dim=-1)
        self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # mug: active이면 spawn 위치, inactive이면 scene 밖
        mug_pos_world = torch.where(is_cup.unsqueeze(1), offscene_pos, obj_pos_world)
        mug_root_state = torch.cat([mug_pos_world, upright_rot, zero_vel], dim=-1)
        self.mug.write_root_state_to_sim(mug_root_state, env_ids=env_ids)

        # ---- 버퍼 초기화 ----
        self.actions[env_ids].zero_()
