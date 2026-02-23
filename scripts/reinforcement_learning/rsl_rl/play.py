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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--blend_vis",
    action="store_true",
    default=False,
    help="Visualize skill blending weights as colored spheres on joints (GUI only).",
)
parser.add_argument("--blend_vis_env", type=int, default=0, help="Environment index to visualize.")
parser.add_argument("--blend_vis_bins", type=int, default=11, help="Number of color bins for blending weights.")
parser.add_argument("--blend_vis_radius", type=float, default=0.025, help="Sphere radius for blend visualization.")
parser.add_argument("--blend_vis_opacity", type=float, default=0.4, help="Sphere opacity for blend visualization.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--no_export",
    action="store_true",
    default=False,
    help="Disable exporting the policy to JIT/ONNX during play.",
)
parser.add_argument(
    "--use_log_cfg",
    action="store_true",
    default=False,
    help="Load env/agent config from the checkpoint log's params/*.yaml before play.",
)
parser.add_argument("--swap_lr", action="store_true", help="Enable left/right swapping for data augmentation.")
parser.add_argument("--swap_lr_prob", type=float, default=0.5, help="Probability to swap each environment per episode.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import re
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from sbm.rl import register_rsl_rl

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict, update_class_from_dict
from isaaclab.utils.io import load_yaml
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx
from sbm.rl import RslRlVecEnvWrapper

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

register_rsl_rl()


def _resolve_pipeline_log_components(task_name: str) -> tuple[str, str]:
    """Resolve <side>/<folder> under pipeline from the registered task config path."""
    task_key = task_name.split(":")[-1].replace("-Play", "")
    fallback_folder = task_key.replace("-", "_")
    try:
        spec = gym.spec(task_key)
        env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(env_cfg_entry, str):
            match = re.search(r"\.pipeline\.(?:gripper|hand)\.(left|right|both)\.([A-Za-z0-9_]+)\.", env_cfg_entry)
            if match:
                return match.group(1), match.group(2)
    except Exception:
        pass
    if "_right" in fallback_folder.lower():
        return "right", fallback_folder
    if "_both" in fallback_folder.lower():
        return "both", fallback_folder
    return "left", fallback_folder


def _unwrap_actions(action):
    if isinstance(action, dict):
        for key in ("actions", "action", "actions_mean", "policy"):
            if key in action:
                return _unwrap_actions(action[key])
        if len(action) == 1:
            return _unwrap_actions(next(iter(action.values())))
    if isinstance(action, (tuple, list)) and len(action) == 1:
        return _unwrap_actions(action[0])
    return action


def _create_blend_markers(num_bins: int, radius: float, opacity: float) -> VisualizationMarkers:
    markers = {}
    steps = max(num_bins - 1, 1)
    for idx in range(num_bins):
        t = idx / steps
        color = (1.0 - t, 0.0, t)
        markers[f"blend_{idx:02d}"] = sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
        )
    cfg = VisualizationMarkersCfg(prim_path="/World/Visuals/SkillBlendWeights", markers=markers)
    return VisualizationMarkers(cfg)


def _get_action_joint_names(env) -> list[str]:
    joint_names: list[str] = []
    for term in env.unwrapped.action_manager._terms.values():
        if hasattr(term, "_joint_names"):
            joint_names.extend(term._joint_names)
    return joint_names


def _map_joint_names_to_body_indices(body_names: list[str], joint_names: list[str]) -> list[int]:
    body_indices: list[int] = []
    for joint_name in joint_names:
        candidates = [
            joint_name.replace("finger_joint", "finger_link"),
            joint_name.replace("_joint", "_link"),
            joint_name.replace("joint", "link"),
            joint_name.replace("_joint", ""),
            joint_name.replace("joint", ""),
        ]
        body_idx = None
        for candidate in candidates:
            if candidate in body_names:
                body_idx = body_names.index(candidate)
                break
        if body_idx is None:
            for candidate in candidates:
                for body_idx, body_name in enumerate(body_names):
                    if body_name.startswith(candidate):
                        break
                else:
                    continue
                break
            else:
                body_idx = 0
        body_indices.append(body_idx)
    return body_indices


def _get_term_action_norm(term) -> float:
    if hasattr(term, "processed_actions"):
        actions = term.processed_actions
    else:
        actions = term.raw_actions
    if actions is None or not hasattr(actions, "norm"):
        return 0.0
    return actions.norm(dim=1).mean().item()


def _log_left_right_metrics(env) -> None:
    env_u = env.unwrapped
    if not hasattr(env_u, "_play_debug_logged"):
        env_u._play_debug_logged = False
    scene_keys = list(env_u.scene.keys())
    try:
        left_arm = env_u.action_manager.get_term("left_arm_action")
        right_arm = env_u.action_manager.get_term("right_arm_action")
        left_hand = env_u.action_manager.get_term("left_hand_action")
        right_hand = env_u.action_manager.get_term("right_hand_action")
        left_arm_norm = _get_term_action_norm(left_arm)
        right_arm_norm = _get_term_action_norm(right_arm)
        left_hand_norm = _get_term_action_norm(left_hand)
        right_hand_norm = _get_term_action_norm(right_hand)
    except Exception:
        left_arm_norm = right_arm_norm = left_hand_norm = right_hand_norm = 0.0

    try:
        if "left_ee_frame" in scene_keys and "right_ee_frame" in scene_keys:
            left_eef = env_u.scene["left_ee_frame"].data.target_pos_w[..., 0, :] - env_u.scene.env_origins
            right_eef = env_u.scene["right_ee_frame"].data.target_pos_w[..., 0, :] - env_u.scene.env_origins
        else:
            body_pos_w = env_u.scene["robot"].data.body_pos_w
            body_names = env_u.scene["robot"].data.body_names
            left_name = "openarm_left_hand" if "openarm_left_hand" in body_names else "openarm_left_ee_tcp"
            right_name = "openarm_right_hand" if "openarm_right_hand" in body_names else "openarm_right_ee_tcp"
            left_idx = body_names.index(left_name)
            right_idx = body_names.index(right_name)
            left_eef = body_pos_w[:, left_idx] - env_u.scene.env_origins
            right_eef = body_pos_w[:, right_idx] - env_u.scene.env_origins
        if "cup" in scene_keys:
            object_pos = env_u.scene["cup"].data.root_pos_w - env_u.scene.env_origins
        else:
            object_pos = env_u.scene["object"].data.root_pos_w - env_u.scene.env_origins
        if "cup2" in scene_keys:
            object2_pos = env_u.scene["cup2"].data.root_pos_w - env_u.scene.env_origins
        else:
            object2_pos = env_u.scene["object2"].data.root_pos_w - env_u.scene.env_origins
        offset = getattr(getattr(env_u, "cfg", None), "grasp2g_target_offset", (0.0, 0.0, 0.0))
        if isinstance(offset, (list, tuple)) and len(offset) == 3:
            offset = torch.tensor(offset, device=object_pos.device)
            object_pos = object_pos + offset
            object2_pos = object2_pos + offset
        left_dist = (object_pos - left_eef).norm(dim=1).mean().item()
        right_dist = (object2_pos - right_eef).norm(dim=1).mean().item()
        if (left_dist == 0.0 or right_dist == 0.0) and not env_u._play_debug_logged:
            env_u._play_debug_logged = True
            print(f"[PLAY_DEBUG] scene_keys={scene_keys}", flush=True)
            ee_label = "frame" if "left_ee_frame" in env_u.scene else "body"
            print(f"[PLAY_DEBUG] ee_source={ee_label}", flush=True)
            print(
                f"[PLAY_DEBUG] left_eef={left_eef[0].tolist()} right_eef={right_eef[0].tolist()} "
                f"obj={object_pos[0].tolist()} obj2={object2_pos[0].tolist()} "
                f"offset={offset.tolist() if hasattr(offset, 'tolist') else offset}",
                flush=True,
            )
    except Exception as exc:
        left_dist = right_dist = 0.0
        if not env_u._play_debug_logged:
            env_u._play_debug_logged = True
            print(f"[PLAY_DEBUG] scene_keys={scene_keys}", flush=True)
            try:
                body_names = env_u.scene["robot"].data.body_names
                print(f"[PLAY_DEBUG] body_names_sample={body_names[:8]}", flush=True)
            except Exception:
                print("[PLAY_DEBUG] body_names_sample=unavailable", flush=True)
            if "left_ee_frame" in scene_keys:
                frame = env_u.scene["left_ee_frame"]
                print(
                    f"[PLAY_DEBUG] left_ee_frame target_pos_w shape="
                    f"{getattr(frame.data.target_pos_w, 'shape', None)}",
                    flush=True,
                )
            print(f"[PLAY_DEBUG] dist_exception={repr(exc)}", flush=True)

    print(
        "[PLAY] left_arm_norm={:.3f} right_arm_norm={:.3f} "
        "left_hand_norm={:.3f} right_hand_norm={:.3f} "
        "left_dist={:.3f} right_dist={:.3f}".format(
            left_arm_norm, right_arm_norm, left_hand_norm, right_hand_norm, left_dist, right_dist
        )
    , flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # CHECKPOINT SEARCH ROOT RULE:
    #   <sbm_root>/log/rsl_rl/pipeline/<left|right|both>/<task_dir_name>
    # side/folder are auto-resolved from task's env_cfg_entry_point (pipeline module path).
    side_dir, task_dir_name = _resolve_pipeline_log_components(train_task_name)
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_root_path = os.path.join(sbm_root, "log", "rsl_rl", "pipeline", side_dir, task_dir_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Default behavior: resolve checkpoint from log_root_path + load_run/load_checkpoint pattern.
        # Typical run_dir is "testN".
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # optionally restore env/agent configs from the training log
    if args_cli.use_log_cfg:
        env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
        if os.path.exists(env_cfg_path):
            def _load_yaml_with_slices(path: str) -> dict:
                try:
                    return load_yaml(path)
                except Exception:
                    import yaml

                    class _SliceLoader(yaml.FullLoader):
                        pass

                    def _slice_constructor(loader, node):
                        seq = loader.construct_sequence(node)
                        return slice(*seq)

                    _SliceLoader.add_constructor(
                        "tag:yaml.org,2002:python/object/apply:builtins.slice", _slice_constructor
                    )
                    with open(path, "r", encoding="utf-8") as f:
                        return yaml.load(f, Loader=_SliceLoader)

            def _prune_none_type_overrides(obj, data: dict) -> dict:
                pruned = {}
                for k, v in data.items():
                    if not (hasattr(obj, k) or (isinstance(obj, dict) and k in obj)):
                        continue
                    obj_mem = obj[k] if isinstance(obj, dict) else getattr(obj, k)
                    if isinstance(v, dict) and obj_mem is not None:
                        pruned[k] = _prune_none_type_overrides(obj_mem, v)
                        continue
                    if obj_mem is None and v is not None:
                        continue
                    pruned[k] = v
                return pruned

            env_cfg_data = _load_yaml_with_slices(env_cfg_path)
            env_cfg_data = _prune_none_type_overrides(env_cfg, env_cfg_data)
            update_class_from_dict(env_cfg, env_cfg_data)
        else:
            print(f"[WARN] env config not found at: {env_cfg_path}")

        agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
        if os.path.exists(agent_cfg_path):
            agent_cfg_data = _load_yaml_with_slices(agent_cfg_path)
            agent_cfg_data = _prune_none_type_overrides(agent_cfg, agent_cfg_data)
            update_class_from_dict(agent_cfg, agent_cfg_data)
        else:
            print(f"[WARN] agent config not found at: {agent_cfg_path}")

    # re-apply CLI overrides after optional log cfg load
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    swap_prob = args_cli.swap_lr_prob if args_cli.swap_lr else 0.0
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        swap_prob=swap_prob,
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    blend_vis_enabled = (
        args_cli.blend_vis
        and not args_cli.headless
        and hasattr(policy_nn, "act_inference_hrl")
        and hasattr(policy_nn, "skill_names")
    )
    if blend_vis_enabled:
        blend_markers = _create_blend_markers(
            num_bins=args_cli.blend_vis_bins,
            radius=args_cli.blend_vis_radius,
            opacity=args_cli.blend_vis_opacity,
        )
        joint_names = _get_action_joint_names(env)
        body_names = env.unwrapped.scene["robot"].data.body_names
        body_indices = _map_joint_names_to_body_indices(body_names, joint_names)
        num_bins = args_cli.blend_vis_bins
        env_index = min(max(args_cli.blend_vis_env, 0), env.unwrapped.num_envs - 1)
    else:
        blend_markers = None
        body_indices = None
        num_bins = None
        env_index = None

    # export policy to onnx/jit
    if not args_cli.no_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    step_count = 0
    # prepare joint logging (exclude finger joints)
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    non_finger_joint_ids = [i for i, name in enumerate(joint_names) if "finger" not in name]
    episode_counts = torch.zeros(env.unwrapped.num_envs, dtype=torch.int64, device=env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            actions = _unwrap_actions(actions)
            masks = None
            if blend_vis_enabled:
                output = policy_nn.act_inference_hrl(obs)
                masks = output.get("masks")
            # env stepping
            obs, _, dones, extras = env.step(actions)
            if step_count % 200 == 0:
                _log_left_right_metrics(env)
            step_count += 1
            # print non-finger joint positions at episode end
            done_mask = dones.squeeze(-1).to(dtype=torch.bool)
            if done_mask.any():
                done_env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                if isinstance(extras, dict) and "terminal_joint_pos" in extras and "terminal_env_ids" in extras:
                    term_ids = extras["terminal_env_ids"].to(device=done_env_ids.device)
                    term_pos = extras["terminal_joint_pos"]
                    # align terminal positions with done env ids
                    idx_map = {int(env_id): i for i, env_id in enumerate(term_ids.tolist())}
                    keep = [idx_map.get(int(env_id), None) for env_id in done_env_ids.tolist()]
                    valid = [i for i, k in enumerate(keep) if k is not None]
                    if valid:
                        sel = torch.tensor([keep[i] for i in valid], device=term_pos.device, dtype=torch.long)
                        joint_pos = term_pos[sel][:, non_finger_joint_ids].detach().cpu()
                        env_ids_for_print = [done_env_ids[i].item() for i in valid]
                    else:
                        joint_pos = robot.data.joint_pos[done_env_ids][:, non_finger_joint_ids].detach().cpu()
                        env_ids_for_print = done_env_ids.tolist()
                else:
                    joint_pos = robot.data.joint_pos[done_env_ids][:, non_finger_joint_ids].detach().cpu()
                    env_ids_for_print = done_env_ids.tolist()
                for idx, env_id in enumerate(env_ids_for_print):
                    episode_counts[env_id] += 1
                    print(
                        f"[EP{int(episode_counts[env_id])}] env={env_id} non_finger_joint_pos="
                        f"{joint_pos[idx].tolist()}"
                    )
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if blend_vis_enabled and masks is not None:
                robot = env.unwrapped.scene["robot"]
                positions = robot.data.body_pos_w[env_index, body_indices]
                weights = masks[env_index]
                if weights.shape[0] >= 2:
                    blend_ratio = weights[1].clamp(0.0, 1.0)
                else:
                    blend_ratio = weights[0].clamp(0.0, 1.0)
                marker_indices = (blend_ratio * (num_bins - 1)).round().to(dtype=torch.int64)
                blend_markers.visualize(
                    translations=positions.detach().cpu(),
                    marker_indices=marker_indices.detach().cpu(),
                )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
