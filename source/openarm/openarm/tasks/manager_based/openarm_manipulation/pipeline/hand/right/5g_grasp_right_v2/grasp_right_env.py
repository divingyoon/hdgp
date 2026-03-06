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

"""환경 클래스: 5g_grasp_right_v2

컨트롤 방식: Geometric Fabrics (DEXTRAH 연계)
  - palm pose 6D → Fabrics → arm 7 DOF joint target
  - finger: palm_dist 기반 자동 보간 (action 없음)
    palm이 approach_trigger_dist일 때 OPEN, 0m일 때 GRASP로 선형 보간

물체: cup / primitive 중 설정에 따라 사용

리워드:
  1. hand_to_object: palm_center가 물체 파지점에 가까워질수록 보상
  2. object_to_goal: 물체 → 목표 위치 (approach_trigger 시 활성)
  3. lift: 물체 수직 들기 (approach_trigger 시 활성)
  4. finger_curl_reg: proximity-gated 파지 자세 보상
  5. palm_orient: 손바닥 법선이 물체를 향하도록 유도
"""

from __future__ import annotations

import math
import os as _os
import sys
from pathlib import Path
import torch
from collections.abc import Sequence

# Prefer vendored FABRICS under hdgp/source/FABRICS/src even when this module is imported directly.
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "source":
        _vendored_fabrics_src = _parent / "FABRICS" / "src"
        if _vendored_fabrics_src.exists():
            _vendored_path = str(_vendored_fabrics_src)
            if _vendored_path not in sys.path:
                sys.path.insert(0, _vendored_path)
        break

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

# Fabrics imports (DEXTRAH 연계)
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
    HAND_START_POSE,
    HAND_GRASP_POSE,
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
    ARM_START_POSE,
    OBJECT_GOAL_POS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_preset import FABRIC_HAND_BODY_NAMES, LEFT_ARM_REST_JOINT_POS
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경.

    액션: 7D
      - [0:6] palm pose (x, y, z, ez, ey, ex), 정규화 [-1, 1]
      - [6]   finger (열림=-1, 닫힘=+1)

    Fabrics: arm(7 DOF)를 palm pose로 제어, hand(20 DOF)는 직접 보간.
    설정에 따라 cup/primitive 중 하나 또는 둘 다 활성화할 수 있다.
    """

    cfg: GraspRightEnvCfg

    @staticmethod
    def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply for wxyz tensors."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @staticmethod
    def _quat_from_euler_zyx_wxyz(ez: torch.Tensor, ey: torch.Tensor, ex: torch.Tensor) -> torch.Tensor:
        """Build wxyz quaternion from intrinsic ZYX Euler angles."""
        hz = 0.5 * ez
        hy = 0.5 * ey
        hx = 0.5 * ex
        cz, sz = torch.cos(hz), torch.sin(hz)
        cy, sy = torch.cos(hy), torch.sin(hy)
        cx, sx = torch.cos(hx), torch.sin(hx)
        qw = cz * cy * cx + sz * sy * sx
        qx = cz * cy * sx - sz * sy * cx
        qy = cz * sy * cx + sz * cy * sx
        qz = sz * cy * cx - cz * sy * sx
        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return torch.nn.functional.normalize(quat, dim=-1)

    def __init__(self, cfg: GraspRightEnvCfg, render_mode: str | None = None, **kwargs):
        # palm_dist_buf는 _setup_geometric_fabrics 이전에 필요하므로 먼저 초기화
        # 실제 값은 super().__init__ 이후 갱신되므로 임시로 큰 값으로 설정
        self._palm_dist_buf_initialized = False
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
        # Hand poses (DEXTRAH 방식)
        # ----------------------------------------------------------------
        self.grasp_pose = to_torch(HAND_GRASP_POSE, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 20)

        # reward 계산용 (20,)
        self.grasp_pose_target = to_torch(HAND_GRASP_POSE, device=self.device)  # (20,)

        # PCA action 범위 (DEXTRAH 방식)
        self.hand_pca_mins = to_torch(HAND_PCA_MINS, device=self.device)  # (5,)
        self.hand_pca_maxs = to_torch(HAND_PCA_MAXS, device=self.device)  # (5,)

        # ----------------------------------------------------------------
        # 로봇 시작 자세: 손가락 열린 상태 (컵 소환 영역 관통 방지)
        # ----------------------------------------------------------------
        arm_start = to_torch(ARM_START_POSE,   device=self.device)   # (7,)
        hand_start = to_torch(HAND_START_POSE, device=self.device)   # (20,)
        robot_start = torch.cat([arm_start, hand_start], dim=0)      # (27,)
        self.robot_start_joint_pos = robot_start.unsqueeze(0).repeat(self.num_envs, 1).contiguous()

        # ----------------------------------------------------------------
        # 왼팔 고정 자세
        # ----------------------------------------------------------------
        left_vals = [
            LEFT_ARM_REST_JOINT_POS.get(self.robot.joint_names[idx], 0.0)
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
        self.pregrasp_offset = to_torch(
            [self.cfg.pregrasp_offset_x, self.cfg.pregrasp_offset_y, self.cfg.pregrasp_offset_z],
            device=self.device,
        )
        ez = torch.tensor([math.radians(self.cfg.pregrasp_orient_offset_ez_deg)], device=self.device)
        ey = torch.tensor([math.radians(self.cfg.pregrasp_orient_offset_ey_deg)], device=self.device)
        ex = torch.tensor([math.radians(self.cfg.pregrasp_orient_offset_ex_deg)], device=self.device)
        self.pregrasp_orient_offset_quat = self._quat_from_euler_zyx_wxyz(ez, ey, ex).squeeze(0)

        # ----------------------------------------------------------------
        # 중간값 버퍼
        # ----------------------------------------------------------------
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.pregrasp_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.pregrasp_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.pregrasp_target_x_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.hand_pos = torch.zeros(self.num_envs, 7, 3, device=self.device)  # 7 bodies × 3D
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_x_pos = torch.zeros(self.num_envs, 3, device=self.device)   # palm +X 방향 마커
        self.fingertip_pos = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        # grasp_target: _compute_intermediate_values에서 계산, _pre_physics_step에서 참조
        self.grasp_target = torch.zeros(self.num_envs, 3, device=self.device)
        # palm_dist_buf: 이전 스텝의 palm_dist 캐시 (_pre_physics_step에서 자동 닫힘 계산용)
        # 초기값: approach_trigger_dist (완전히 열린 상태로 시작)
        self.palm_dist_buf = torch.full(
            (self.num_envs,), self.cfg.approach_trigger_dist, device=self.device
        )
        self.pregrasp_dist_buf = torch.full(
            (self.num_envs,), self.cfg.pregrasp_activate_dist, device=self.device
        )
        self.pregrasp_orient_align_buf = torch.zeros(self.num_envs, device=self.device)
        self.grasp_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 물체 유형 추적: 0=cup, 1=primitive (둘 다 활성 시 에피소드마다 랜덤)
        # ----------------------------------------------------------------
        self.active_object_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ----------------------------------------------------------------
        # 컵 초기 XY 위치 (에피소드 시작 시 기록 → displacement penalty 기준)
        # ----------------------------------------------------------------
        self.cup_initial_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # cup_tipping 판정용 cos threshold
        self._cup_tipping_cos = math.cos(math.radians(self.cfg.cup_tipping_max_deg))

        # ----------------------------------------------------------------
        # Fabrics 초기화 (arm 제어용)
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

        # ----------------------------------------------------------------
        # Fabrics cspace attractor를 GRASP_POSE로 고정 (DEXTRAH curled_q에 해당)
        # PCA attractor는 에이전트가 5D action으로 직접 제어
        # ----------------------------------------------------------------
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, NUM_ARM_DOF:] = self.grasp_pose
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        # ----------------------------------------------------------------
        # Hand FK taskmap (Fabrics URDF 기준, 8 bodies)
        # ----------------------------------------------------------------
        robot_dir_name = "openarm_tesollo"
        robot_name = "openarm_tesollo"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path, FABRIC_HAND_BODY_NAMES, self.num_envs, self.device
        )

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        if (not self.cfg.enable_cup) and (not self.cfg.enable_primitives):
            raise ValueError("At least one of enable_cup or enable_primitives must be True.")

        self.robot = Articulation(self.cfg.robot_cfg)
        self.cup = RigidObject(self.cfg.cup_cfg) if self.cfg.enable_cup else None
        self.primitive = None
        self.table = RigidObject(self.cfg.table_cfg)

        self.scene.articulations["robot"] = self.robot
        if self.cup is not None:
            self.scene.rigid_objects["cup"] = self.cup
        self.scene.rigid_objects["table"] = self.table

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # DEXTRAH와 동일하게 source prim을 기준으로 env 복제하여
        # USD/scene 메모리 중복을 줄인다.
        self.scene.clone_environments(copy_from_source=True)

        if self.cfg.enable_primitives:
            self._setup_random_primitives()

    def _setup_random_primitives(self) -> None:
        """Create one random primitive asset per env (DEXTRAH-style object bank spawn)."""
        primitives_root = _os.path.join(OPENARM_ROOT_DIR, "../../../../../../assets/primitives/USD")
        primitives_root = _os.path.normpath(primitives_root)
        sub_dirs = sorted(
            d for d in _os.listdir(primitives_root) if _os.path.isdir(_os.path.join(primitives_root, d))
        )
        if len(sub_dirs) == 0:
            raise ValueError(f"No primitive folders found under: {primitives_root}")

        sampled_ids = torch.randint(0, len(sub_dirs), (self.num_envs,), device=self.device).cpu().tolist()
        for i in range(self.num_envs):
            primitive_name = sub_dirs[sampled_ids[i]]
            usd_path = _os.path.join(primitives_root, primitive_name, f"{primitive_name}.usd")
            prim_path = f"/World/envs/env_{i}/Primitive/primitive_{i}_{primitive_name}"
            object_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=self.cfg.primitive_cfg.spawn.scale,
                    articulation_props=self.cfg.primitive_cfg.spawn.articulation_props,
                    rigid_props=self.cfg.primitive_cfg.spawn.rigid_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(self.cfg.primitive_cfg.init_state.pos),
                    rot=tuple(self.cfg.primitive_cfg.init_state.rot),
                ),
            )
            _ = RigidObject(object_cfg)

        primitive_regex_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Primitive/.*",
            spawn=None,
        )
        self.primitive = RigidObject(primitive_regex_cfg)
        self.scene.rigid_objects["primitive"] = self.primitive

    # ------------------------------------------------------------------
    # Geometric Fabrics 초기화 (DEXTRAH _setup_geometric_fabrics 기반)
    # ------------------------------------------------------------------
    def _setup_geometric_fabrics(self) -> None:
        warp_cache_dir = self.device[-1]  # GPU index
        initialize_warp(warp_cache_dir)

        print("=== GraspRightEnv: Creating Fabrics world ===")
        world_filename = "open_tesollo_boxes"
        max_objects_per_env = self.cfg.fabrics_max_objects_per_env
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
        # hand_pca_targets: 5D PCA (0으로 고정 — hand는 cspace attractor로 제어)
        self.hand_pca_targets = torch.zeros(self.num_envs, 5, device=self.device)
        # palm_pose_targets: 6D palm pose (ez, ey, ex 순서 euler_zyx)
        self.palm_pose_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # Fabric cspace damping gain (고정값, ADR 없음)
        self.fabric_damping_gain = 10.0 * torch.ones(self.num_envs, 1, device=self.device)

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

        # ---- 액션 파싱: 6D palm + 5D PCA (DEXTRAH 방식) ----
        palm_action = actions[:, :6]   # (N, 6), 정규화 [-1, 1]
        pca_action  = actions[:, 6:]   # (N, 5), 정규화 [-1, 1]

        # Palm pose 스케일 (정규화 → 실제 workspace)
        palm_pose = scale(palm_action, self.palm_mins, self.palm_maxs)  # (N, 6)
        self.palm_pose_targets.copy_(palm_pose)

        # PCA hand 스케일 → Fabrics PCA attractor 직접 제어
        # cspace attractor는 HAND_GRASP_POSE 고정 (default_config 불변)
        hand_pca = scale(pca_action, self.hand_pca_mins, self.hand_pca_maxs)  # (N, 5)
        self.hand_pca_targets.copy_(hand_pca)

        # ---- Fabrics 실행 (arm + hand 제어) ----
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

    def _apply_action(self) -> None:
        # 오른팔+오른손: Fabrics가 arm + hand (27D) 모두 제어
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
        # 물체 위치: cup/primitive 활성 상태에 따라 선택
        if self.cup is not None and self.primitive is not None:
            cup_pos = self.cup.data.root_pos_w - self.scene.env_origins           # (N, 3)
            cup_rot = self.cup.data.root_quat_w                                    # (N, 4)
            primitive_pos = self.primitive.data.root_pos_w - self.scene.env_origins  # (N, 3)
            primitive_rot = self.primitive.data.root_quat_w                        # (N, 4)
            use_primitive = (self.active_object_type == 1).unsqueeze(1)            # (N, 1)
            self.object_pos = torch.where(use_primitive.expand(-1, 3), primitive_pos, cup_pos)
            self.object_rot = torch.where(use_primitive.expand(-1, 4), primitive_rot, cup_rot)
        elif self.cup is not None:
            self.object_pos = self.cup.data.root_pos_w - self.scene.env_origins
            self.object_rot = self.cup.data.root_quat_w
        else:
            self.object_pos = self.primitive.data.root_pos_w - self.scene.env_origins
            self.object_rot = self.primitive.data.root_quat_w

        # 손 위치: Fabrics FK (7 bodies × 3D = 21D)
        # [0]=palm_center, [1]=palm_x, [2:7]=fingertips
        hand_pos_flat, _ = self.hand_points_taskmap(self.fabric_q, None)  # (N, 21)
        all_pos = hand_pos_flat.view(self.num_envs, 7, 3)  # (N, 7, 3)
        self.palm_center_pos = all_pos[:, 0, :]
        self.palm_x_pos      = all_pos[:, 1, :]
        self.fingertip_pos   = all_pos[:, 2:, :]
        self.hand_pos = all_pos

        # grasp_target 및 palm_dist_buf 업데이트
        # → 다음 스텝의 _pre_physics_step에서 자동 닫힘 계산에 사용
        self.grasp_target = self.object_pos.clone()
        self.grasp_target[:, 2] += self.cfg.object_grasp_z_offset
        self.palm_dist_buf.copy_((self.palm_center_pos - self.grasp_target).norm(dim=-1))
        self.pregrasp_dist_buf.copy_((self.palm_center_pos - self.pregrasp_pos).norm(dim=-1))

        target_x_local = torch.zeros(self.num_envs, 3, device=self.device)
        target_x_local[:, 0] = 1.0
        self.pregrasp_target_x_dir = torch.nn.functional.normalize(
            quat_apply(self.pregrasp_quat, target_x_local), dim=-1
        )
        palm_x_dir = torch.nn.functional.normalize(self.palm_x_pos - self.palm_center_pos, dim=-1)
        self.pregrasp_orient_align_buf.copy_((palm_x_dir * self.pregrasp_target_x_dir).sum(dim=-1))

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
            self.hand_pos.view(self.num_envs, -1),                      # (N, 21)
            # 물체 상태
            self.object_pos,                                            # (N, 3)
            self.object_rot,                                            # (N, 4)
            # 목표 위치
            self.object_goal,                                           # (N, 3)
            self.object_init_pos,                                       # (N, 3)
            (self.pregrasp_pos - self.palm_center_pos),                 # (N, 3)
            self.pregrasp_target_x_dir,                                 # (N, 3)
            # 마지막 액션
            self.actions,                                               # (N, 11)
            # Fabrics 상태 (arm 추론용)
            self.fabric_q,                                              # (N, 27)
            self.fabric_qd,                                             # (N, 27)
        ], dim=-1)  # (N, 150)

        return {"policy": obs, "critic": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- 1. hand_to_object (palm_center only) ----
        # grasp_target, palm_dist: _compute_intermediate_values에서 이미 계산됨
        grasp_target = self.grasp_target   # (N, 3)
        palm_dist = self.palm_dist_buf     # (N,)
        pregrasp_dist = self.pregrasp_dist_buf

        pregrasp_reward = self.cfg.pregrasp_reach_weight * torch.exp(
            -self.cfg.pregrasp_reach_sharpness * pregrasp_dist
        )
        pregrasp_orient_align = self.pregrasp_orient_align_buf
        orient_error = (1.0 - pregrasp_orient_align).clamp(min=0.0)
        orient_gate = (pregrasp_dist < self.cfg.pregrasp_orient_activate_dist).float()
        pregrasp_orient_reward = orient_gate * self.cfg.pregrasp_orient_weight * torch.exp(
            -self.cfg.pregrasp_orient_sharpness * orient_error
        )

        hand_to_obj_reward = self.cfg.hand_to_object_weight * torch.exp(
            -self.cfg.hand_to_object_sharpness * palm_dist
        )

        # ---- 2. object_to_goal ----
        goal_dist = (self.object_pos - self.object_goal).norm(dim=-1)  # (N,)
        obj_to_goal_reward = self.cfg.object_to_goal_weight * torch.exp(
            -self.cfg.object_to_goal_sharpness * goal_dist
        )

        # ---- 3. lift ----
        lift_error = (self.object_pos[:, 2] - self.object_goal[:, 2]).abs()  # (N,)
        lift_reward = self.cfg.lift_weight * torch.exp(
            -self.cfg.lift_sharpness * lift_error
        )

        # ---- 4. finger_curl_reg (DEXTRAH 방식) ----
        # cspace attractor = HAND_GRASP_POSE 고정 → 손가락이 자연히 파지 방향으로 당겨짐
        # 에이전트는 PCA action으로 손가락 조정
        # proximity-gated: palm이 컵에 가까울 때 grasp_pose에 가까울수록 보상
        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]  # (N, 20)
        proximity = torch.exp(-self.cfg.curl_proximity_sharpness * palm_dist)

        grasp_error = (hand_q - self.grasp_pose_target.unsqueeze(0)).pow(2).mean(dim=-1)
        grasp_formed = self._compute_grasp_formed_mask(palm_dist, grasp_error)

        finger_grasp_reward = self.cfg.finger_grasp_weight * proximity * torch.exp(
            -self.cfg.finger_grasp_sharpness * grasp_error
        )
        finger_open_reg = torch.zeros_like(finger_grasp_reward)  # 제거 (cspace attractor가 대체)

        curl_reg = finger_grasp_reward

        # ---- 5. palm orientation reward ----
        # palm +X (손바닥 법선) 방향이 컵을 향하도록 유도
        palm_x_dir = torch.nn.functional.normalize(
            self.palm_x_pos - self.palm_center_pos, dim=-1
        )  # (N, 3): 손바닥 법선 방향 (world frame)

        palm_to_cup = torch.nn.functional.normalize(
            grasp_target - self.palm_center_pos, dim=-1
        )  # (N, 3): palm → 컵 파지 중심 방향

        # 내적: 1=완벽 정렬, 0=수직, -1=반대 방향
        align = (palm_x_dir * palm_to_cup).sum(dim=-1)  # (N,)
        palm_orient_reward = self.cfg.palm_orient_weight * align

        # ---- 바이너리 단계 게이트 ----
        # approach_trigger: palm이 컵 근처(approach_trigger_dist)에 도달 → lift/goal 활성화
        # cup_lifted 조건 제거: 파지 이전에 lift/goal 탐색 기회를 부여
        #   - 자동 닫힘(palm_dist 기반)과 결합: palm 접근 → 손가락 자동 닫힘 → 파지 → 들기
        #   - cup_tipping termination(60도)이 비정상적인 밀기 행동을 억제
        approach_trigger = (
            (pregrasp_dist < self.cfg.pregrasp_activate_dist)
            & (pregrasp_orient_align > self.cfg.pregrasp_orient_success_cos)
        ).float()
        grasp_trigger = approach_trigger  # cup_lifted 조건 제거
        if self.cfg.grasp_only_mode:
            goal_scale = self.cfg.grasp_only_goal_reward_scale
            lift_scale = self.cfg.grasp_only_lift_reward_scale
        else:
            goal_scale = 1.0
            lift_scale = 1.0

        # ---- 합산 ----
        total = (
            pregrasp_reward                         # 초기 reference 유도
            + pregrasp_orient_reward                # orientation reference 유도
            + hand_to_obj_reward                    # 항상 활성: 접근 유도
            + curl_reg                              # proximity-gated: 파지 자세 유도
            + self.cfg.grasp_stability_weight * grasp_formed.float()
            + palm_orient_reward                    # 항상 활성: 방향 정렬 유도
            + goal_scale * grasp_trigger * obj_to_goal_reward    # approach_trigger 시 활성
            + lift_scale * grasp_trigger * lift_reward           # approach_trigger 시 활성
        )

        # 로깅
        self.extras["pregrasp_reward"] = pregrasp_reward.mean()
        self.extras["pregrasp_dist"] = pregrasp_dist.mean()
        self.extras["pregrasp_orient_reward"] = pregrasp_orient_reward.mean()
        self.extras["pregrasp_orient_align"] = pregrasp_orient_align.mean()
        self.extras["grasp_formed"] = grasp_formed.float().mean()
        self.extras["grasp_error"] = grasp_error.mean()
        self.extras["grasp_hold_steps"] = self.grasp_hold_steps.float().mean()
        self.extras["hand_to_object_reward"] = hand_to_obj_reward.mean()
        self.extras["obj_to_goal_reward"] = obj_to_goal_reward.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["finger_grasp_reward"] = finger_grasp_reward.mean()
        self.extras["finger_open_reg"] = finger_open_reg.mean()
        self.extras["finger_curl_reg"] = curl_reg.mean()
        self.extras["palm_orient_reward"] = palm_orient_reward.mean()
        self.extras["palm_align"] = align.mean()
        self.extras["palm_dist"] = palm_dist.mean()
        self.extras["goal_dist"] = goal_dist.mean()
        self.extras["object_z"] = self.object_pos[:, 2].mean()
        self.extras["approach_trigger"] = approach_trigger.mean()
        self.extras["grasp_trigger"] = grasp_trigger.mean()
        # 자동 닫힘 정도 (0=완전열림, 1=완전닫힘)
        finger_t = (1.0 - (palm_dist / self.cfg.approach_trigger_dist).clamp(0.0, 1.0))
        self.extras["finger_t"] = finger_t.mean()
        if self.cup is not None and self.primitive is not None:
            self.extras["primitive_ratio"] = (self.active_object_type == 1).float().mean()
            self.extras["cup_ratio"] = (self.active_object_type == 0).float().mean()
        else:
            self.extras["primitive_ratio"] = torch.zeros((), device=self.device)
            self.extras["cup_ratio"] = torch.ones((), device=self.device) if self.cup is not None else torch.zeros((), device=self.device)

        return total

    def _compute_grasp_formed_mask(self, palm_dist: torch.Tensor, grasp_error: torch.Tensor) -> torch.Tensor:
        height_ok = (self.object_pos[:, 2] - self.cfg.object_spawn_z) < self.cfg.grasp_success_max_height_delta
        return (
            (palm_dist < self.cfg.grasp_success_palm_dist)
            & (grasp_error < self.cfg.grasp_success_hand_error)
            & height_ok
        )

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]
        grasp_error = (hand_q - self.grasp_pose_target.unsqueeze(0)).pow(2).mean(dim=-1)
        grasp_formed = self._compute_grasp_formed_mask(self.palm_dist_buf, grasp_error)

        self.grasp_hold_steps = torch.where(
            grasp_formed,
            self.grasp_hold_steps + 1,
            torch.zeros_like(self.grasp_hold_steps),
        )
        grasp_success = self.grasp_hold_steps >= self.cfg.grasp_success_hold_steps

        # 물체가 작업 영역 밖으로 나갔을 때 종료
        out_x = (self.object_pos[:, 0] < 0.05) | (self.object_pos[:, 0] > 0.85)
        out_y = (self.object_pos[:, 1] < -0.55) | (self.object_pos[:, 1] > 0.25)
        fallen = self.object_pos[:, 2] < 0.18  # 테이블 위 최소 높이

        # 컵 기울기 초과 종료 (cup local z축 vs world z축의 cos 값으로 판단)
        # quat_apply: (N,4) × (N,3) → (N,3)
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)          # (N, 3)
        tilt_cos = cup_z_world[:, 2]                                  # world +Z 성분
        tipped = tilt_cos < self._cup_tipping_cos

        terminated = out_x | out_y | fallen | tipped
        if self.cfg.terminate_on_grasp_success:
            terminated = terminated | grasp_success
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        self.extras["grasp_success"] = grasp_success.float().mean()

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

        # ---- 물체 유형 배정 ----
        if self.cup is not None and self.primitive is not None:
            self.active_object_type[env_ids] = torch.randint(
                0, 2, (n,), device=self.device, dtype=torch.long
            )
            is_cup = (self.active_object_type[env_ids] == 0)  # (n,) bool
            is_primitive = ~is_cup
        elif self.cup is not None:
            self.active_object_type[env_ids] = 0
            is_cup = torch.ones(n, device=self.device, dtype=torch.bool)
            is_primitive = torch.zeros(n, device=self.device, dtype=torch.bool)
        else:
            self.active_object_type[env_ids] = 1
            is_cup = torch.zeros(n, device=self.device, dtype=torch.bool)
            is_primitive = torch.ones(n, device=self.device, dtype=torch.bool)

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
        self.object_init_pos[env_ids] = obj_pos_local

        # ---- object-relative pregrasp target 생성 (DemoGrasp-style) ----
        pregrasp_noise = torch.stack(
            [
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_x,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_y,
                (torch.rand(n, device=self.device) - 0.5) * 2.0 * self.cfg.pregrasp_noise_z,
            ],
            dim=1,
        )
        pregrasp_pos_local = obj_pos_local + self.pregrasp_offset.unsqueeze(0) + pregrasp_noise
        self.pregrasp_pos[env_ids] = pregrasp_pos_local
        ez = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.radians(self.cfg.pregrasp_orient_noise_ez_deg)
        ey = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.radians(self.cfg.pregrasp_orient_noise_ey_deg)
        ex = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.radians(self.cfg.pregrasp_orient_noise_ex_deg)
        noise_quat = self._quat_from_euler_zyx_wxyz(ez, ey, ex)
        offset_quat = self.pregrasp_orient_offset_quat.unsqueeze(0).repeat(n, 1)
        rel_quat = self._quat_mul_wxyz(offset_quat, noise_quat)
        object_quat = upright_rot
        self.pregrasp_quat[env_ids] = self._quat_mul_wxyz(object_quat, rel_quat)

        # displacement penalty 기준: 에피소드 시작 시 spawn XY 기록 (local frame)
        self.cup_initial_xy[env_ids] = obj_pos_local[:, :2]

        # ---- inactive 물체는 scene 밖 아래로 이동 ----
        offscene_pos = self.scene.env_origins[env_ids].clone()
        offscene_pos[:, 2] -= 10.0  # env local z=-10m (테이블 아래)

        upright_rot = torch.zeros(n, 4, device=self.device)
        upright_rot[:, 0] = 1.0   # w=1 (identity quaternion)
        zero_vel = torch.zeros(n, 6, device=self.device)

        # cup: active이면 spawn 위치, inactive이면 scene 밖
        if self.cup is not None:
            cup_pos_world = torch.where(is_cup.unsqueeze(1), obj_pos_world, offscene_pos)
            cup_root_state = torch.cat([cup_pos_world, upright_rot, zero_vel], dim=-1)
            self.cup.write_root_state_to_sim(cup_root_state, env_ids=env_ids)

        # primitive: active이면 spawn 위치, inactive이면 scene 밖
        if self.primitive is not None:
            primitive_pos_world = torch.where(is_primitive.unsqueeze(1), obj_pos_world, offscene_pos)
            primitive_root_state = torch.cat([primitive_pos_world, upright_rot, zero_vel], dim=-1)
            self.primitive.write_root_state_to_sim(primitive_root_state, env_ids=env_ids)

        # ---- 버퍼 초기화 ----
        self.actions[env_ids].zero_()
        # palm_dist_buf: reward 계산용, 리셋 시 큰 값으로 초기화
        self.palm_dist_buf[env_ids] = self.cfg.approach_trigger_dist
        self.pregrasp_dist_buf[env_ids] = self.cfg.pregrasp_activate_dist
        self.pregrasp_orient_align_buf[env_ids] = 0.0
        self.grasp_hold_steps[env_ids] = 0
