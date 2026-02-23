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

import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR

# Hand joint names
# Split thumb and pinky into separate action heads to avoid coupled optimization.
LEFT_THUMB_JOINTS = [f"lj_dg_1_{joint}" for joint in range(1, 5)]
LEFT_PINKY_JOINTS = [f"lj_dg_5_{joint}" for joint in range(1, 5)]

# Fingers 2-4: independently controlled (no synergy action).
NON_THUMB_JOINTS_LEFT = [f"lj_dg_{finger}_{joint}" for finger in range(2, 5) for joint in range(1, 5)]
NON_THUMB_JOINTS_RIGHT = [f"rj_dg_{finger}_{joint}" for finger in range(2, 5) for joint in range(1, 5)]

# Full hand joints (reference)
LEFT_HAND_JOINTS = [f"lj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]
RIGHT_HAND_JOINTS = [f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]

from .. import mdp
from ..lift_left_env_cfg import Lift5gLeftEnvCfg


@configclass
class OpenArmLift5gLeftEnvCfg(Lift5gLeftEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/openarm_tesollo_t3.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.0, 0.0, -0.25],
                rot=[1.0, 0.0, 0.0, 0.0],
                joint_pos={
                    "openarm_left_joint1": -0.5,
                    "openarm_left_joint2": -0.5,
                    "openarm_left_joint3": 0.6,
                    "openarm_left_joint4": 0.7,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": -1.0,
                    "openarm_right_joint1": 0.5,
                    "openarm_right_joint2": 0.5,
                    "openarm_right_joint3": -0.6,
                    "openarm_right_joint4": 0.7,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 1.0,
                    # Left fingers 2,3,4 (independent)
                    "lj_dg_[2-4]_.*": 0.0,
                    # Left thumb (1)
                    "lj_dg_1_1": 0.0,
                    "lj_dg_1_2": 1.571,   # [0.0, 1.571] max open
                    "lj_dg_1_3": 0.0,
                    "lj_dg_1_4": 0.0,
                    # Left pinky (5)
                    "lj_dg_5_1": 0.0,
                    "lj_dg_5_2": 0.0,
                    "lj_dg_5_3": 0.0,
                    "lj_dg_5_4": 0.0,
                    # Right fingers 2,3,4 (independent)
                    "rj_dg_[2-4]_.*": 0.0,
                    # Right thumb (1)
                    "rj_dg_1_1": 0.0,
                    "rj_dg_1_2": -1.571,  # [-1.571, 0.0] max open
                    "rj_dg_1_3": 0.0,
                    "rj_dg_1_4": 0.0,
                    # Right pinky (5)
                    "rj_dg_5_1": 0.0,
                    "rj_dg_5_2": 0.0,
                    "rj_dg_5_3": 0.0,
                    "rj_dg_5_4": 0.0,
                },
            ),
            actuators={
                "openarm_arm": ImplicitActuatorCfg(
                    joint_names_expr=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
                    stiffness=400.0,
                    damping=80.0,
                ),
                "openarm_gripper": ImplicitActuatorCfg(
                    joint_names_expr=["lj_dg_.*", "rj_dg_.*"],
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        cup_usd = f"{OPENARM_ROOT_DIR}/usds/openarm_bimanual/cup.usd"
        self.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0.1, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(1.0, 1.0, 1.2),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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

        self.scene.cup2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, -0.1, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=cup_usd,
                scale=(1.0, 1.0, 1.2),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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

        # Contact sensors are available in t3 robot USD (sensor_link bodies).
        self.scene.left_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tesollo_left_.*_sensor_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"],
            history_length=3,
            track_air_time=False,
        )
        self.scene.right_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tesollo_right_.*_sensor_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup2"],
            history_length=3,
            track_air_time=False,
        )

        # Left-only action order: left_arm/hand/thumb/pinky
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            scale=0.2,
            use_default_offset=True,
        )
        # Left hand fingers 2-4: independent 12-DoF joint control.
        self.actions.left_hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=NON_THUMB_JOINTS_LEFT,
            scale={
                "lj_dg_2_1": 0.2, "lj_dg_2_2": 0.8, "lj_dg_2_3": 0.6, "lj_dg_2_4": 0.6,
                "lj_dg_3_1": 0.2, "lj_dg_3_2": 0.8, "lj_dg_3_3": 0.6, "lj_dg_3_4": 0.6,
                "lj_dg_4_1": 0.2, "lj_dg_4_2": 0.8, "lj_dg_4_3": 0.6, "lj_dg_4_4": 0.6,
            },
            clip={
                "lj_dg_2_1": (-0.1, 0.1), "lj_dg_2_2": (0.0, 2.0), "lj_dg_2_3": (0.0, 1.571), "lj_dg_2_4": (0.0, 1.571),
                "lj_dg_3_1": (-0.1, 0.1), "lj_dg_3_2": (0.0, 2.0), "lj_dg_3_3": (0.0, 1.571), "lj_dg_3_4": (0.0, 1.571),
                "lj_dg_4_1": (-0.1, 0.1), "lj_dg_4_2": (0.0, 2.0), "lj_dg_4_3": (0.0, 1.571), "lj_dg_4_4": (0.0, 1.571),
            },
            use_default_offset=True,
        )
        self.actions.left_thumb_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_THUMB_JOINTS,
            scale={
                "lj_dg_1_1": 0.8,
                "lj_dg_1_2": 1.0,
                "lj_dg_1_3": 0.8,
                "lj_dg_1_4": 0.8,
            },
            clip={
                # 1_1: no strict preference; keep broad anatomical range.
                "lj_dg_1_1": (-0.8901179, 0.3839724),
                # Thumb requested ranges
                "lj_dg_1_2": (0.0, 1.571),
                "lj_dg_1_3": (-1.571, 0.0),
                "lj_dg_1_4": (-1.571, 0.0),
            },
            use_default_offset=True,
        )
        self.actions.left_pinky_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_PINKY_JOINTS,
            scale={
                "lj_dg_5_1": 0.1,
                "lj_dg_5_2": 0.4,
                "lj_dg_5_3": 0.8,
                "lj_dg_5_4": 0.8,
            },
            clip={
                # Pinky requested ranges
                "lj_dg_5_1": (-0.1, 0.1),
                "lj_dg_5_2": (-0.6109, 0.0),
                "lj_dg_5_3": (0.0, 1.571),
                "lj_dg_5_4": (0.0, 1.571),
            },
            use_default_offset=True,
        )

        self.commands.object_pose.body_name = "ll_dg_ee"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class OpenArmLift5gLeftEnvCfg_PLAY(OpenArmLift5gLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = True
