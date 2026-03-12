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
  - finger: palm_dist 기반 자동 보간 (action 없음, A안)
    t = 1 - clamp(palm_dist / approach_trigger_dist, 0, 1)
    interp = (1-t)*HAND_START_POSE + t*HAND_GRASP_POSE
    Fabrics 실행 후 fabric_q의 hand 부분(index 7~26)을 override

물체: cup / primitive 중 설정에 따라 사용

리워드 (DEXTRAH와 동일 구조: 모든 항 항상 활성, 게이트 없음):
  1. hand_to_object: palm_center → 파지점 접근 유도
  2. palm_orient: 손바닥 법선이 컵을 향하도록 유도 (side-approach 전용)
  3. finger_curl_reg: 보간 목표(interp_hand) 이탈 패널티 (음수 weight)
  4. object_to_goal: 물체 → 목표 위치
  5. lift: 물체 수직 들기
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
from .grasp_adr import GraspADR
from .grasp_right_constants import (
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    HAND_START_POSE,
    HAND_GRASP_POSE,
    ARM_START_POSE,
    OBJECT_GOAL_POS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
)
from .grasp_right_utils import scale, to_torch


class GraspRightEnv(DirectRLEnv):
    """OpenArm+Teosllo 오른손 파지 환경.

    액션: 6D
      - [0:6] palm pose (x, y, z, ez, ey, ex), 정규화 [-1, 1]

    Fabrics: arm(7 DOF)를 palm pose로 제어.
    hand(20 DOF)는 palm_dist 기반 자동 보간 (HAND_START_POSE → HAND_GRASP_POSE).
    설정에 따라 cup/primitive 중 하나 또는 둘 다 활성화할 수 있다.
    """

    cfg: GraspRightEnvCfg

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
        # Hand poses (palm_dist 기반 자동 보간)
        # ----------------------------------------------------------------
        self.hand_open_pose  = to_torch(HAND_START_POSE, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 20)
        self.grasp_pose      = to_torch(HAND_GRASP_POSE, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # (N, 20)

        # reward 계산용 (20,)
        self.grasp_pose_target = to_torch(HAND_GRASP_POSE, device=self.device)  # (20,)

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
        # hand_pos: Fabrics FK (7 bodies × 3D): [0]=palm_center, [1]=palm_x, [2:7]=fingertips
        self.hand_pos = torch.zeros(self.num_envs, 7, 3, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_x_pos = torch.zeros(self.num_envs, 3, device=self.device)   # palm +X 방향 마커
        self.fingertip_pos = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        # grasp_target: _compute_intermediate_values에서 계산, _pre_physics_step에서 참조
        self.grasp_target = torch.zeros(self.num_envs, 3, device=self.device)
        # interp_hand: palm_dist 기반 보간 결과 (N, 20), _pre_physics_step에서 계산
        self.interp_hand = self.hand_open_pose.clone()
        # palm_dist_buf: 이전 스텝의 palm_dist 캐시 (_pre_physics_step에서 자동 닫힘 계산용)
        # 초기값: approach_trigger_dist (완전히 열린 상태로 시작)
        self.palm_dist_buf = torch.full(
            (self.num_envs,), self.cfg.approach_trigger_dist, device=self.device
        )
        # finger_t_buf: ratchet mechanism — 에피소드 내 최대 닫힘 t값 누적
        # palm이 위로 올라가도 이전 최대값 유지 → 들어올리는 동안 파지 유지
        self.finger_t_buf = torch.zeros(self.num_envs, device=self.device)

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
        # ADR 초기화 (GraspADR: DEXTRAH DextrahADR 이식)
        # ----------------------------------------------------------------
        if cfg.enable_adr:
            self.grasp_adr = GraspADR(
                custom_cfg=cfg.adr_custom_cfg,
                num_increments=cfg.adr_num_increments,
                increment_interval=cfg.adr_increment_interval,
                trigger_threshold=cfg.adr_trigger_threshold,
            )
        else:
            self.grasp_adr = None

        # ----------------------------------------------------------------
        # Fabrics 초기화 (arm 제어용)
        # ----------------------------------------------------------------
        self._setup_geometric_fabrics()

        # ----------------------------------------------------------------
        # Fabrics cspace attractor를 HAND_START_POSE (열린 자세)로 설정
        # 손가락은 palm_dist 기반 자동 보간으로 fabric_q를 직접 override하므로
        # cspace attractor는 arm 제어에만 실질적으로 영향을 미침
        # ----------------------------------------------------------------
        cspace_default = self.open_tesollo_fabric.default_config.clone()
        cspace_default[:, :NUM_ARM_DOF] = arm_start.unsqueeze(0).expand(self.num_envs, -1)
        cspace_default[:, NUM_ARM_DOF:] = self.hand_open_pose
        self.open_tesollo_fabric.default_config.copy_(cspace_default)

        # ----------------------------------------------------------------
        # 초기 액션: ARM_START_POSE에서의 FK palm pose를 역스케일
        # self.actions.zero_() → workspace 중심(z=0.40m) 타겟 → arm이 위로 당겨짐
        # arm_start_action으로 초기화하면 Fabrics가 ARM_START_POSE 부근에 머뭄
        # ----------------------------------------------------------------
        _q_start = self.robot_start_joint_pos[:1]   # (1, 27)
        _palm_start = self.open_tesollo_fabric.get_palm_pose(_q_start, "euler_zyx")  # (1, 6)
        _range = self.palm_maxs - self.palm_mins     # (6,)
        self.arm_start_action = (
            (_palm_start[0] - self.palm_mins) / _range * 2.0 - 1.0
        ).clamp(-1.0, 1.0)  # (6,)

        # ----------------------------------------------------------------
        # Hand FK taskmap (Fabrics URDF 기준, 8 bodies)
        # ----------------------------------------------------------------
        robot_dir_name = "openarm_tesollo"
        robot_name = "openarm_tesollo"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        fabric_hand_body_names = [
            "palm_center",           # [0]: palm 중심 (h2o 타겟)
            "palm_x",                # [1]: palm +X 방향 마커 (0.25m 손바닥 법선 방향)
            "tesollo_right_rl_dg_1_4",  # [2]: thumb tip
            "tesollo_right_rl_dg_2_4",  # [3]: index tip
            "tesollo_right_rl_dg_3_4",  # [4]: middle tip
            "tesollo_right_rl_dg_4_4",  # [5]: ring tip
            "tesollo_right_rl_dg_5_4",  # [6]: pinky tip
        ]
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path, fabric_hand_body_names, self.num_envs, self.device
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
        # Remove only FABRICS-level table repulsion for v1.
        # The sim table collision remains active through PhysX.
        world_filename = "open_tesollo_boxes_no_table"
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

        # ---- 액션 파싱: 6D palm only ----
        palm_action = actions[:, :6]   # (N, 6), 정규화 [-1, 1]

        # Palm pose 스케일 (정규화 → 실제 workspace)
        palm_pose = scale(palm_action, self.palm_mins, self.palm_maxs)  # (N, 6)
        self.palm_pose_targets.copy_(palm_pose)

        # hand_pca_targets: 0 고정 (손가락은 자동 보간으로 제어)
        self.hand_pca_targets.zero_()

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

        # ---- palm_dist 기반 손가락 자동 보간 (Ratchet Mechanism) ----
        # palm_dist_buf: 이전 스텝에서 계산된 palm_center ~ grasp_target 거리
        # t_new = 1 - clamp(palm_dist / trigger, 0, 1)
        # ratchet: finger_t_buf = max(finger_t_buf, t_new)
        #   → palm이 위로 올라가도 한번 닫힌 손가락은 열리지 않음
        #   → 들어올리는 동안 파지 유지
        t_new = (1.0 - (self.palm_dist_buf / self.cfg.approach_trigger_dist).clamp(0.0, 1.0))  # (N,)
        torch.max(self.finger_t_buf, t_new, out=self.finger_t_buf)  # ratchet, in-place
        t = self.finger_t_buf.unsqueeze(1)  # (N, 1)
        # in-place 보간: interp_hand 버퍼 재사용
        self.interp_hand.copy_(self.hand_open_pose)
        self.interp_hand.mul_(1.0 - t)
        self.interp_hand.add_(t * self.grasp_pose)
        self.fabric_q[:, NUM_ARM_DOF:] = self.interp_hand
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

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
        self.grasp_target.copy_(self.object_pos)
        self.grasp_target[:, 2] += self.cfg.object_grasp_z_offset
        self.palm_dist_buf.copy_((self.palm_center_pos - self.grasp_target).norm(dim=-1))

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
            # 마지막 액션
            self.actions,                                               # (N, 6)
            # Fabrics 상태 (arm 추론용)
            self.fabric_q,                                              # (N, 27)
            self.fabric_qd,                                             # (N, 27)
        ], dim=-1)  # (N, 145)

        return {"policy": obs, "critic": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # ---- ADR: DEXTRAH 동일 구조 ----
        # lift_weight     : 5→0  (초기 강하게, ADR 진행에 따라 감소)
        # goal_sharpness  : 15→20 (점점 정밀 요구)
        # curl_weight     : -2→-1 (패널티 완화)
        # object_to_goal_weight: 5.0 고정 (DEXTRAH 동일, ADR 대상 아님)
        if self.grasp_adr is not None:
            lift_weight        = self.grasp_adr.get_param("reward_weights", "lift_weight")
            finger_curl_weight = self.grasp_adr.get_param("reward_weights", "finger_curl_weight")
            goal_sharpness     = self.grasp_adr.get_param("sharpness", "object_to_goal_sharpness")
        else:
            lift_weight        = self.cfg.lift_weight
            finger_curl_weight = self.cfg.finger_curl_weight
            goal_sharpness     = self.cfg.object_to_goal_sharpness

        # ---- 공통 참조 ----
        grasp_target = self.grasp_target  # (N, 3)
        palm_dist = self.palm_dist_buf    # (N,)

        # ---- 1. hand_to_object: palm_center → grasp_target ----
        # DEXTRAH: max over 6 hand points / v1: palm_dist (side-approach 전용)
        hand_to_obj_reward = self.cfg.hand_to_object_weight * torch.exp(
            -self.cfg.hand_to_object_sharpness * palm_dist
        )

        # ---- 2. palm orientation (side-approach 전용) ----
        # 손바닥 법선(palm +X)이 컵을 향하도록 유도
        palm_x_dir = torch.nn.functional.normalize(
            self.palm_x_pos - self.palm_center_pos, dim=-1
        )  # (N, 3)
        palm_to_cup = torch.nn.functional.normalize(
            grasp_target - self.palm_center_pos, dim=-1
        )  # (N, 3)
        align = (palm_x_dir * palm_to_cup).sum(dim=-1)  # (N,) ∈ [-1, 1]
        palm_orient_reward = self.cfg.palm_orient_weight * align

        # ---- 3. finger curl (DEXTRAH R_curl 동일 방식) ----
        # w_curl * mean((q_hand - q_interp)²), w_curl < 0
        # DEXTRAH: 고정 q_curled / v1: palm_dist 기반 동적 q_interp (side-approach 전용)
        hand_q = self.robot.data.joint_pos[:, self.hand_dof_indices]  # (N, 20)
        curl_error = (hand_q - self.interp_hand).pow(2).mean(dim=-1)  # (N,)
        curl_reg = finger_curl_weight * curl_error  # ADR: -2→-1 (완화)

        # ---- 4. object_to_goal (DEXTRAH: weight 5.0 고정, sharpness ADR로 증가) ----
        goal_dist = (self.object_pos - self.object_goal).norm(dim=-1)  # (N,)
        obj_to_goal_reward = self.cfg.object_to_goal_weight * torch.exp(
            -goal_sharpness * goal_dist  # ADR: 15→20
        )

        # ---- 5. lift (DEXTRAH: weight ADR로 5→0 감소, sharpness 8.5 고정) ----
        lift_error = (self.object_pos[:, 2] - self.object_goal[:, 2]).abs()  # (N,)
        lift_reward = lift_weight * torch.exp(  # ADR: 5→0
            -self.cfg.lift_sharpness * lift_error
        )

        # ---- 합산 (DEXTRAH 동일 구조) ----
        total = (
            hand_to_obj_reward   # 접근 유도
            + palm_orient_reward # 방향 정렬 (side-approach 전용)
            + curl_reg           # 파지 자세 패널티
            + obj_to_goal_reward # 목표 위치 이동
            + lift_reward        # 수직 들기
        )

        # ---- ADR increment 트리거 ----
        # 컵이 lift_adr_threshold(5cm) 이상 들린 env 비율이 threshold(10%) 초과 시 increment
        if self.grasp_adr is not None:
            lift_success_ratio = (
                self.object_pos[:, 2] > self.cfg.object_spawn_z + self.cfg.lift_adr_threshold
            ).float().mean()
            self.grasp_adr.maybe_increment(lift_success_ratio)

        # 로깅
        self.extras["hand_to_object_reward"] = hand_to_obj_reward.mean()
        self.extras["palm_orient_reward"] = palm_orient_reward.mean()
        self.extras["palm_align"] = align.mean()
        self.extras["curl_reg"] = curl_reg.mean()
        self.extras["obj_to_goal_reward"] = obj_to_goal_reward.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["palm_dist"] = palm_dist.mean()
        self.extras["goal_dist"] = goal_dist.mean()
        self.extras["object_z"] = self.object_pos[:, 2].mean()
        self.extras["finger_t"] = self.finger_t_buf.mean()
        self.extras["adr_lift_weight"] = torch.tensor(lift_weight, device=self.device)
        self.extras["adr_goal_sharpness"] = torch.tensor(goal_sharpness, device=self.device)
        self.extras["adr_curl_weight"] = torch.tensor(finger_curl_weight, device=self.device)
        self.extras["adr_progress"] = torch.tensor(
            self.grasp_adr.progress if self.grasp_adr is not None else 0.0,
            device=self.device,
        )
        if self.cup is not None and self.primitive is not None:
            self.extras["primitive_ratio"] = (self.active_object_type == 1).float().mean()
            self.extras["cup_ratio"] = (self.active_object_type == 0).float().mean()
        else:
            self.extras["primitive_ratio"] = torch.zeros((), device=self.device)
            self.extras["cup_ratio"] = (
                torch.ones((), device=self.device)
                if self.cup is not None
                else torch.zeros((), device=self.device)
            )

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

        # 컵 기울기 초과 종료 (cup local z축 vs world z축의 cos 값으로 판단)
        # quat_apply: (N,4) × (N,3) → (N,3)
        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cup_z_world = quat_apply(self.object_rot, z_local)          # (N, 3)
        tilt_cos = cup_z_world[:, 2]                                  # world +Z 성분
        tipped = tilt_cos < self._cup_tipping_cos

        terminated = out_x | out_y | fallen | tipped
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
        self.actions[env_ids] = self.arm_start_action.unsqueeze(0).expand(n, -1)
        # palm_dist_buf: reward 계산용, 리셋 시 큰 값으로 초기화 (손가락 완전 열림)
        self.palm_dist_buf[env_ids] = self.cfg.approach_trigger_dist
        # interp_hand: 열린 자세로 초기화
        self.interp_hand[env_ids] = self.hand_open_pose[env_ids]
        # finger_t_buf: ratchet 초기화 (에피소드 시작 = 완전 열림)
        self.finger_t_buf[env_ids] = 0.0
