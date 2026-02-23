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

"""Lift-style left 5g environment following 2g_grasp_left_v1 reward structure."""

from dataclasses import MISSING
import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp


@configclass
class Lift5gLeftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING
    left_contact_sensor: ContactSensorCfg = MISSING
    right_contact_sensor: ContactSensorCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    # Left-only policy actions.
    left_arm_action: ActionTerm = MISSING
    left_hand_action: ActionTerm = MISSING
    left_thumb_action: ActionTerm = MISSING
    left_pinky_action: ActionTerm = MISSING


# Create larger marker config for better visibility
GOAL_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
GOAL_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)

LEFT_CONTACT_LINKS = [
    "tesollo_left_ll_dg_1_3",
    "tesollo_left_ll_dg_1_4",
    "tesollo_left_ll_dg_2_3",
    "tesollo_left_ll_dg_2_4",
    "tesollo_left_ll_dg_3_3",
    "tesollo_left_ll_dg_3_4",
    "tesollo_left_ll_dg_4_3",
    "tesollo_left_ll_dg_4_4",
    "tesollo_left_ll_dg_5_3",
    "tesollo_left_ll_dg_5_4",
]


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.3),
            pos_y=(0.1, 0.2),
            pos_z=(0.3, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=GOAL_MARKER_CFG,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cup")})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        left_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        left_hand_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_hand_action"})
        left_thumb_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_thumb_action"})
        left_pinky_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_pinky_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cup_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (0.2, 0.2),
                "z": (0.0, 0.0),
                "yaw": (math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup"),
        },
    )

    reset_cup2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.25),
                "y": (-0.2, -0.2),
                "z": (0.0, 0.0),
                "yaw": (0, 0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cup2"),
        },
    )

    # Keep right side fixed so left-only policy does not need right-side actions.
    lock_right_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*", "rj_dg_.*"]),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=8.0,
    )
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance_fine,
        params={"std": 0.065, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=10.0,
    )

    end_effector_orientation = RewTerm(
        func=mdp.eef_z_perpendicular_object_z,
        params={"std": 0.3, "eef_link_name": "ll_dg_ee", "object_cfg": SceneEntityCfg("cup")},
        weight=4.0,
    )
    finger_closing_pregrasp = RewTerm(
        func=mdp.finger_grasp_reward,
        params={"std": 2.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=4.0,
    )
    fingertip_cup_facing_pregrasp = RewTerm(
        func=mdp.finger_tip_orientation_reward,
        params={"std": 0.5, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=4.0,
    )

    # v1의 grasp 계열을 contact-sensor 기반으로 대체.
    grasp_contact_persistence = RewTerm(
        func=mdp.contact_persistence_reward,
        params={
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
            "min_contacts": 3,
            "contact_threshold": 0.02,
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "ll_dg_ee",
            "require_thumb_contact": True,
        },
        weight=5.0,
    )
    grasp_contact_coverage = RewTerm(
        func=mdp.contact_finger_coverage_reward,
        params={
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "ll_dg_ee",
            "contact_threshold": 0.02,
            "min_fingers_bonus": 4,
            "bonus_scale": 1.0,
            "require_thumb_contact": True,
        },
        weight=5.0,
    )
    grasp_strict_success = RewTerm(
        func=mdp.strict_grasp_lift_success,
        params={
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "ll_dg_ee",
            "contact_threshold": 0.02,
            "required_fingers": 4,
            "minimal_height": 0.04,
            "hold_steps": 6,
            "require_thumb_contact": True,
        },
        weight=10.0,
    )

    pregrasp_contact_penalty = RewTerm(
        func=mdp.pregrasp_contact_penalty,
        params={
            "sensor_cfg": SceneEntityCfg("left_contact_sensor"),
            "object_cfg": SceneEntityCfg("cup"),
            "eef_link_name": "ll_dg_ee",
            "contact_threshold": 0.02,
            "max_allowed_contacts": 1,
        },
        weight=-6.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cup")},
        weight=15.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose", "object_cfg": SceneEntityCfg("cup")},
        weight=10.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.04, "command_name": "object_pose", "object_cfg": SceneEntityCfg("cup")},
        weight=10.0,
    )

    object_displacement = RewTerm(
        func=mdp.object_displacement_penalty,
        params={"object_cfg": SceneEntityCfg("cup"), "threshold": 0.005},
        weight=-5.0,
    )

    thumb_reaching_pose = RewTerm(
        func=mdp.thumb_reaching_pose_reward,
        params={"std": 1.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.5,
    )

    pinky_reaching_pose = RewTerm(
        func=mdp.pinky_reaching_pose_reward,
        params={"std": 1.0, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.5,
    )
    synergy_reaching_pose = RewTerm(
        func=mdp.synergy_reaching_pose_reward,
        params={"std": 1.5, "object_cfg": SceneEntityCfg("cup"), "eef_link_name": "ll_dg_ee"},
        weight=0.5,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "lj_dg_.*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cup_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")})
    cup_tipping = DoneTerm(func=mdp.cup_tipped, params={"asset_cfg": SceneEntityCfg("cup"), "max_tilt_deg": 35.0})


@configclass
class CurriculumCfg:
    # Disable curriculum terms for now; keep fixed reward weights.
    pass


@configclass
class Lift5gLeftEnvCfg(ManagerBasedRLEnvCfg):
    task_name: str = "pickplace_5g_left_v1"
    curriculum_stage: int = 0
    mask_inactive_arm_actions: bool = True
    grasp2g_target_offset: tuple[float, float, float] = (0.01, -0.06, 0.08)
    reach_dynamic_z_high: float = 0.25
    reach_dynamic_xy_hi: float = 0.10
    reach_dynamic_xy_lo: float = 0.03
    reach_dynamic_xy_gate: float = 0.03
    reach_dynamic_z_descent_rate: float = 0.005
    reach_displacement_free_threshold: float = 0.015
    reach_displacement_suppress_scale: float = 0.03
    # DexPour stage trigger (lambda): approach complete threshold.
    dexpour_approach_threshold: float = 0.08
    # DexPour stage trigger (mu): secure grasp contact condition.
    dexpour_grasp_min_contacts: int = 4
    dexpour_contact_threshold: float = 0.02
    dexpour_require_thumb_contact: bool = True
    # Optional smoothing for nu trigger; default is strict binary trigger.
    dexpour_use_soft_nu_gate: bool = False
    debug_stage_triggers: bool = True
    debug_stage_triggers_interval: int = 50
    reach_switch_threshold: float = 0.05
    reach_switch_hold_steps: int = 2
    # Soft gate for grasp/contact rewards to avoid hard 0/1 dead-zone near transition.
    reach_soft_gate_near: float = 0.02
    reach_soft_gate_far: float = 0.10
    # Separate grasp activation gate: require closer EE-target distance before finger closing.
    grasp_switch_threshold: float = 0.025
    grasp_switch_hold_steps: int = 4
    grasp_soft_gate_near: float = 0.012
    grasp_soft_gate_far: float = 0.05
    # Grasp gate safety: enable finger-closing reward only after orientation + low-push pre-grasp.
    grasp_soft_prefactor: float = 0.2
    grasp_orientation_std: float = 0.2
    grasp_orientation_gate_min_reward: float = 0.25
    grasp_orientation_gate_full_reward: float = 0.75
    grasp_displacement_free_threshold: float = 0.01
    grasp_displacement_suppress_scale: float = 0.015
    # Transfer stage gate (DexPour stage-3 analog): trigger after stable lift.
    transfer_soft_height_lo: float = 0.05
    transfer_soft_height_hi: float = 0.10
    transfer_switch_height: float = 0.10
    transfer_switch_hold_steps: int = 4
    require_filtered_contact_matrix: bool = True
    # Debug visualization
    debug_approach_target_vis: bool = True
    debug_fingertip_vis: bool = True
    debug_fingertip_vis_interval: int = 5

    scene: Lift5gLeftSceneCfg = Lift5gLeftSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.commands.object_pose.debug_vis = False

        self.observations.policy.concatenate_terms = True

        self.sim.physx.bounce_threshold_velocity = 0.01
        # 256 환경용 (v1와 동일)
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 64 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**24
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.friction_correlation_distance = 0.00625
