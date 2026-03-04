# OpenArm + Teosllo right arm pose fabric
# Based on kuka_allegro_pose_fabric.py
# 7 DOF OpenArm right arm + 20 DOF Teosllo right hand = 27 DOF total

import torch

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion


class OpenArmTeoslloPoseFabric(BaseFabric):
    """
    Fabric for OpenArm right arm (7 DOF) + Teosllo right hand (20 DOF) = 27 DOF total.
    Action space: 6D palm pose + 5D hand PCA = 11D (same as DEXTRAH paper).

    Joint order in URDF (27 revolute joints):
      [0-6]  openarm_right_joint1~7     (arm)
      [7-10] rj_dg_1_1~4               (thumb)
      [11-14] rj_dg_2_1~4              (index)
      [15-18] rj_dg_3_1~4              (middle)
      [19-22] rj_dg_4_1~4              (ring)
      [23-26] rj_dg_5_1~4              (pinky)
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True):
        fabric_params_filename = "openarm_tesollo_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        robot_dir_name = "openarm_tesollo"
        robot_name = "openarm_tesollo"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)

        self.load_robot(robot_dir_name, robot_name, batch_size)

        # Default cspace config (27 DOF):
        # Arm: natural working pose for right OpenArm
        # Hand: slightly curled (Kuka-Allegro 방식, 0.5~0.75 구부러짐과 동일)
        #   thumb _2 음수=curl, 나머지 손가락 _2 양수=curl
        #   robot_start_joint_pos 및 curled_q와 일치
        default_config = torch.tensor([
            # OpenArm right arm joint1~7
            1.0,  -0.1,   0.0,  0.5,  0.0,  0.0,  0.0,
            # Teosllo thumb (rj_dg_1_1~4):
            #   _1(X): 0.0 (neutral abduction)
            #   _2(Z): -1.0 (opposition curl, 한계 -π, 32%)
            #   _3,_4: 0.5 (distal flex; PC3 최대 시 _2=-1.5, _3=0.5+0.7=1.2)
            0.0,  -1.0,  0.5,  0.5,
            # Index (rj_dg_2_1~4):
            #   _1(X, 외전): 0.0  _2(Y, curl): 0.7(한계2.007, 35%)  _3,_4: 0.5
            0.0,   0.7,  0.5,  0.5,
            # Middle (rj_dg_3_1~4):
            #   _2(Y, curl): 0.7(한계1.955, 36%)
            0.0,   0.7,  0.5,  0.5,
            # Ring (rj_dg_4_1~4):
            #   _2(Y, curl): 0.7(한계1.902, 37%)
            0.0,   0.7,  0.5,  0.5,
            # Pinky (rj_dg_5_1~4):
            #   _1(Z, 굽힘관절!): 0.0 (사용자 확인: 파지 자세에서 _1=0)
            #   _2(X, 외전): 0.0  _3: 0.7  _4: 0.5
            0.0,   0.0,  0.7,  0.5,
        ], device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self._pca_matrix = None

        self.construct_fabric()

        # Allocate palm pose target tensor (b x 12): 3D origin + 9D rotation matrix
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        self._native_palm_pose_target = None

    # ------------------------------------------------------------------
    # Fabric construction methods
    # ------------------------------------------------------------------

    def add_joint_limit_repulsion(self):
        joints = self.urdfpy_robot.joints
        upper_joint_limits = []
        lower_joint_limits = []
        for j in joints:
            if j.joint_type == 'revolute':
                upper_joint_limits.append(j.limit.upper)
                lower_joint_limits.append(j.limit.lower)

        # Upper limit repulsion
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

        # Lower limit repulsion
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

    def add_cspace_attractor(self, is_forcing):
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        if not is_forcing:
            fabric_name = "cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, fabric_name, fabric)
        else:
            fabric_name = "forcing_cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['forcing_cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)

        self.add_fabric(taskmap_name, fabric_name, fabric)

    def add_hand_fabric(self):
        """
        PCA action space for Teosllo 20-DOF hand.
        Maps 5D PCA → 20D hand joint space.

        Joint order in hand block (indices 7-26):
          [0-3]  thumb  rj_dg_1_1~4
          [4-7]  index  rj_dg_2_1~4
          [8-11] middle rj_dg_3_1~4
          [12-15] ring  rj_dg_4_1~4
          [16-19] pinky rj_dg_5_1~4

        PC1: global proximal+middle flexion  (_2=0.38, _3=0.45 — Kuka 데이터 기반)
        PC2: global distal flexion           (_4=0.55 지배 — Kuka 데이터 기반)
        PC3: proximal펼침 + middle굽힘 diff  (_2 음수 + _3 양수 — Kuka 데이터 기반)
        PC4: thumb opposition 전담           (Kuka에서 thumb 담당 PC)
        PC5: spread + complex differential   (Kuka 데이터 기반)
        """
        # 5 x 20 PCA matrix  (rows = PCA components, cols = hand joints)
        # Col order: [th1,th2,th3,th4, idx1,idx2,idx3,idx4,
        #             mid1,mid2,mid3,mid4, rng1,rng2,rng3,rng4,
        #             pnk1,pnk2,pnk3,pnk4]
        #
        # ─── 설계 원칙 ──────────────────────────────────────────────────────────
        # index/middle/ring: Kuka Allegro PCA 계수 직접 매핑
        #   Allegro [_0,_1,_2,_3] → Teosllo [_1,_2,_3,_4]  (구조 동일)
        #   _3/_4(distal)가 PC1~PC5 여러 곳에 분산 → 파지 중 굽힘 가능
        # pinky: pnk_1(Z,굽힘)/pnk_2(X,외전) 한계 제약 → 0으로 고정
        #   pnk_3/pnk_4 = rng_2/rng_3 (ring flex 계수 재사용)
        # thumb: 구조 다름 (Allegro CMC/MCP/PIP/DIP → Teosllo abduction/Z/X/X)
        #   thm_1(X, abduction) = 0  (별도)
        #   thm_2(Z, curl, 음수=curl) = -Kuka_thm0  (부호 반전)
        #   thm_3(X, proximal flex)  = Kuka_thm1
        #   thm_4(X, distal flex)    = (Kuka_thm2 + Kuka_thm3) / 2
        # ─────────────────────────────────────────────────────────────────────
        pca_matrix = torch.tensor([
            # PC1 – global proximal + middle flexion (전체 파지 모드)
            # 손가락이 전반적으로 굽혀지는 패턴 (Kuka 데이터 기반)
            # _3(middle flex)이 0.447로 크게 기여 → 파지 중 손가락 전체 굽힘
            # thm2: +0.015(Kuka) → -0.15(수정): PC1 활성화 시 엄지도 함께 닫힘
            #   이유: "전체 파지" PC1 사용 시 엄지가 살짝 열리는 문제 수정
            [ 0.000, -0.150,  0.098,  0.044,   # thumb: thm2=-0.150(수정), thm3=0.098, thm4=0.044
              -0.039,  0.379,  0.447,  0.007,  # index: _2=0.379, _3=0.447
               0.002,  0.320,  0.447,  0.052,  # middle
               0.000,  0.298,  0.386,  0.008,  # ring
               0.000,  0.000,  0.386,  0.008], # pinky: _3,_4만 (rng 계수 재사용)

            # PC2 – global distal flexion
            # _4(distal curl)이 0.549~0.579으로 지배 → 끝마디 굽힘 전담 PC
            [ 0.000, -0.030,  0.026,  0.067,   # thumb
              -0.051, -0.130,  0.058,  0.579,  # index: _4=0.579 (distal 지배!)
               0.010, -0.185,  0.054,  0.549,  # middle
               0.000, -0.177,  0.028,  0.482,  # ring
               0.000,  0.000,  0.028,  0.482], # pinky

            # PC3 – proximal펼침 + middle굽힘 differential
            # _2(proximal) 음수 + _3(middle) 양수 → "손가락 중간만 굽히기"
            [ 0.000, -0.002,  0.037, -0.101,   # thumb: opposition 거의 없음
              -0.057, -0.347,  0.334, -0.180,  # index
              -0.044, -0.477,  0.325, -0.152,  # middle
               0.000, -0.458,  0.365, -0.139,  # ring
               0.000,  0.000,  0.365, -0.139], # pinky

            # PC4 – thumb opposition + ring/pinky 보정
            # thm2(Z curl), thm3(proximal), thm4(distal) 모두 기여
            # Kuka에서 PC4가 thumb 전담 PC
            [ 0.000, -0.227, -0.599,  0.229,   # thumb: thm2=-0.227, thm3=-0.599, thm4=(0.703-0.245)/2
               0.023, -0.034,  0.034, -0.027,  # index
               0.023,  0.046,  0.098, -0.001,  # middle
               0.000, -0.014,  0.138,  0.029,  # ring
               0.000,  0.000,  0.138,  0.029], # pinky

            # PC5 – spread + complex differential
            # 다양한 패턴: proximal 펼침 + distal 굽힘 혼합
            # thm2: +0.476(Kuka) → 0.000(수정): PC5 spread 사용 시 엄지 curl이 상쇄되는 문제 수정
            #   이유: PC5+=spread 패턴에서 thm2=+0.476이면 엄지가 강하게 열려
            #         PC4의 thumb curl 효과를 최대 +0.952 상쇄 → 엄지가 안 닫히는 원인
            [ 0.000,  0.000, -0.277, -0.240,   # thumb: thm2=0.000(수정, 무영향), thm3/thm4 유지
              -0.045, -0.472,  0.093,  0.231,  # index
              -0.002,  0.096,  0.125,  0.037,  # middle
               0.000,  0.461,  0.099, -0.081,  # ring
               0.000,  0.000,  0.099, -0.081], # pinky
        ], device=self.device)

        self._pca_matrix = torch.clone(pca_matrix.detach())

        # Pad with zeros for the 7 arm joints (arm joints not controlled via PCA)
        pca_matrix = torch.cat(
            [torch.zeros(pca_matrix.shape[0], 7, device=self.device), pca_matrix], dim=1
        )

        taskmap_name = "pca_hand"
        taskmap = LinearMap(pca_matrix, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        fabric = Attractor(True, self.fabric_params['hand_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "hand_attractor", fabric)

    def add_palm_points_attractor(self):
        """
        7-point palm frame attractor (origin + 6 axis points) for full 6-DOF palm control.
        """
        taskmap_name = "palm"
        control_point_frames = [
            "palm_link",
            "palm_x",  "palm_x_neg",
            "palm_y",  "palm_y_neg",
            "palm_z",  "palm_z_neg",
        ]
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, control_point_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        fabric = Attractor(True, self.fabric_params['palm_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "palm_attractor", fabric)

    def add_body_repulsion(self):
        collision_sphere_frames = self.fabric_params['body_repulsion']['collision_sphere_frames']
        self.collision_sphere_radii = self.fabric_params['body_repulsion']['collision_sphere_radii']

        assert len(collision_sphere_frames) == len(self.collision_sphere_radii), \
            "length of link names does not equal length of radii"

        collision_sphere_pairs = self.fabric_params['body_repulsion']['collision_sphere_pairs']

        collision_matrix = torch.zeros(
            len(collision_sphere_frames), len(collision_sphere_frames),
            dtype=int, device=self.device
        )

        if len(collision_sphere_pairs) == 0:
            collision_link_prefix_pairs = \
                self.fabric_params['body_repulsion']['collision_link_prefix_pairs']
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]
                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])

        for sphere1, sphere2 in collision_sphere_pairs:
            collision_matrix[
                collision_sphere_frames.index(sphere1),
                collision_sphere_frames.index(sphere2)
            ] = 1

        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)

        fabric = BodySphereRepulsion(True, self.fabric_params['body_repulsion'],
                                     self.batch_size, sphere_radius, collision_matrix,
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "repulsion", fabric)

        fabric_geom = BodySphereRepulsion(False, self.fabric_params['body_repulsion'],
                                          self.batch_size, sphere_radius, collision_matrix,
                                          self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        self.base_fabric_repulsion = BaseFabricRepulsion(
            self.fabric_params['body_repulsion'],
            self.batch_size,
            sphere_radius,
            collision_matrix,
            self.device,
        )

    def add_cspace_energy(self):
        taskmap_name = "identity"
        self.add_energy(
            taskmap_name, "euclidean",
            EuclideanEnergy(self.batch_size, self._num_joints, self.device)
        )

    def construct_fabric(self):
        self.add_joint_limit_repulsion()
        self.add_cspace_attractor(False)
        self.add_hand_fabric()
        self.add_palm_points_attractor()
        self.add_body_repulsion()
        self.add_cspace_energy()

    # ------------------------------------------------------------------
    # Runtime methods
    # ------------------------------------------------------------------

    def convert_transform_to_points(self):
        """Convert palm pose target (origin + rotation matrix) to 7×3D control points."""
        palm_transform = torch.zeros(self.batch_size, 4, 4, device=self.device)
        palm_transform[:, 3, 3] = 1.
        palm_transform[:, :3, :3] = torch.transpose(
            self._palm_pose_target[:, 3:].reshape(self.batch_size, 3, 3), 1, 2
        )
        palm_transform[:, :3, 3] = self._palm_pose_target[:, :3]

        def _axis_point(offset_xyz):
            p = torch.zeros(self.batch_size, 4, device=self.device)
            p[:, 3] = 1.
            p[:, 0] = offset_xyz[0]
            p[:, 1] = offset_xyz[1]
            p[:, 2] = offset_xyz[2]
            return p

        palm_targets = torch.zeros(self.batch_size, 7 * 3, device=self.device)
        # Origin
        palm_targets[:, :3] = self._palm_pose_target[:, :3]
        # ±x
        palm_targets[:, 3:6]   = torch.bmm(palm_transform, _axis_point([0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 6:9]   = torch.bmm(palm_transform, _axis_point([-0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        # ±y
        palm_targets[:, 9:12]  = torch.bmm(palm_transform, _axis_point([0., 0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(palm_transform, _axis_point([0., -0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        # ±z
        palm_targets[:, 15:18] = torch.bmm(palm_transform, _axis_point([0., 0., 0.25]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 18:21] = torch.bmm(palm_transform, _axis_point([0., 0., -0.25]).unsqueeze(2)).squeeze(2)[:, :3]
        return palm_targets

    def get_sphere_radii(self):
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        return self.base_fabric_repulsion.collision_status

    def get_palm_pose(self, cspace_position, orientation_convention):
        palm_points, _ = self.get_taskmap("palm")(cspace_position, None)
        palm_origin = palm_points[:, :3]
        x_point = palm_points[:, 3:6]
        y_point = palm_points[:, 9:12]
        z_point = palm_points[:, 15:18]

        x_axis = torch.nn.functional.normalize(x_point - palm_origin, dim=1)
        y_axis = torch.nn.functional.normalize(y_point - palm_origin, dim=1)
        z_axis = torch.nn.functional.normalize(z_point - palm_origin, dim=1)

        rotation_matrix = torch.zeros(self.batch_size, 3, 3, device=self.device)
        rotation_matrix[:, :, 0] = x_axis
        rotation_matrix[:, :, 1] = y_axis
        rotation_matrix[:, :, 2] = z_axis

        if orientation_convention == "euler_zyx":
            orientation = matrix_to_euler(rotation_matrix)
        elif orientation_convention == "quaternion":
            orientation = matrix_to_quaternion(rotation_matrix)[:, [1, 2, 3, 0]]
        else:
            raise ValueError('orientation_convention must be "euler_zyx" or "quaternion"')

        return torch.cat([palm_origin, orientation], dim=-1)

    @property
    def pca_matrix(self):
        return self._pca_matrix

    @pca_matrix.setter
    def pca_matrix(self, pca_matrix):
        self._pca_matrix = pca_matrix

    def set_features(self, hand_target, palm_pose_target, orientation_convention,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator,
                     cspace_damping_gain=None):
        """
        Pass input features to fabric terms.

        Args:
            hand_target:              (B, 5)  PCA hand target
            palm_pose_target:         (B, 6)  [x,y,z, ez,ey,ex] with euler_zyx
                                   or (B, 7)  [x,y,z, qx,qy,qz,qw] with quaternion
            orientation_convention:   "euler_zyx" or "quaternion"
            batched_cspace_position:  (B, 27) current joint positions
            batched_cspace_velocity:  (B, 27) current joint velocities
            object_ids:               Warp array of object mesh IDs
            object_indicator:         Warp array indicating mesh presence
            cspace_damping_gain:      Optional damping gain scalar
        """
        self.fabrics_features["pca_hand"]["hand_attractor"] = hand_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        self._palm_pose_target[:, :3] = palm_pose_target[:, :3]

        if orientation_convention == "euler_zyx":
            assert palm_pose_target.shape[1] == 6, \
                "euler_zyx pose target must be (B, 6)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                euler_to_matrix(palm_pose_target[:, 3:]), 1, 2
            ).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert palm_pose_target.shape[1] == 7, \
                "quaternion pose target must be (B, 7)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                quaternion_to_matrix(palm_pose_target[:, [6, 3, 4, 5]]), 1, 2
            ).reshape(self.batch_size, 9)
        else:
            raise ValueError('orientation_convention must be "euler_zyx" or "quaternion"')

        palm_pose_target_points = self.convert_transform_to_points()

        if self._native_palm_pose_target is None:
            self._native_palm_pose_target = torch.clone(palm_pose_target_points)
        else:
            self._native_palm_pose_target.copy_(palm_pose_target_points)

        try:
            self.fabrics_features["palm"]["palm_attractor"] = self._native_palm_pose_target
            self.get_fabric_term("palm", "palm_attractor").damping_position = \
                self._native_palm_pose_target
        except Exception:
            raise ValueError('No task map "palm" or "palm_attractor"')

        # Compute body sphere positions and velocities
        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

        self.base_fabric_repulsion.calculate_response(
            body_point_pos, body_point_vel, object_ids, object_indicator
        )

        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain
