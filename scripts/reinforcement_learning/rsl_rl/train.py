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

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--able_early_stop", action="store_true", default=False, help="Enable early stopping.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--swap_lr",
    action="store_true",
    default=False,
    help="Enable left/right swapping in the RSL-RL wrapper for data augmentation.",
)
parser.add_argument(
    "--swap_lr_prob", type=float, default=0.5, help="Probability to swap each environment per episode."
)
parser.add_argument(
    "--warmstart_ckpt",
    type=str,
    default=None,
    help="Optional checkpoint for on-demand warmstart rollout at every reset.",
)
parser.add_argument(
    "--warmstart_steps",
    type=int,
    default=0,
    help="Number of warmstart steps to rollout after each reset (requires --warmstart_ckpt).",
)
parser.add_argument("--log_able", action="store_true", default=False, help="Enable episode logging to a file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import re
import shutil
import time
import torch
from datetime import datetime
from typing import Any

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from sbm.rl import register_rsl_rl

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from sbm.rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude


def _unwrap_actions(action):
    """Unwrap nested action dictionaries."""
    if isinstance(action, dict):
        for key in ("actions", "action", "actions_mean", "policy"):
            if key in action:
                return _unwrap_actions(action[key])
        if len(action) == 1:
            return _unwrap_actions(next(iter(action.values())))
    if isinstance(action, (tuple, list)) and len(action) == 1:
        return _unwrap_actions(action[0])
    return action


class WarmStartWrapper(gym.Wrapper):
    """Roll out a warmstart policy for a few steps after each reset."""

    def __init__(self, env: gym.Env, policy, policy_nn, steps: int):
        super().__init__(env)
        self._policy = policy
        self._policy_nn = policy_nn
        self._steps = int(steps)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self._steps <= 0:
            return obs
        for _ in range(self._steps):
            with torch.inference_mode():
                actions = _unwrap_actions(self._policy(obs))
                obs, _, dones, _ = self.env.step(actions)
                if self._policy_nn is not None:
                    self._policy_nn.reset(dones)
        return obs


class EpisodeLogWrapper(gym.Wrapper):
    """Log per-episode reward terms and command stats to a file."""

    def __init__(
        self,
        env: gym.Env,
        log_path: str,
        command_names: tuple[str, str],
        eef_names: tuple[str, str],
    ):
        super().__init__(env)
        self._log_path = log_path
        self._command_names = command_names
        self._eef_names = eef_names
        self._fh = open(self._log_path, "a", encoding="utf-8")
        self._limit_margin_threshold = 0.1
        self._pos_success_thresh = 0.05
        self._ori_success_thresh = 0.2

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        return super().close()

    def step(self, action):
        obs, rew, terminated, truncated, extras = self.env.step(action)
        log_data = extras.get("log", {})
        if log_data:
            env_unwrapped = self.env.unwrapped
            step_count = getattr(env_unwrapped, "common_step_counter", None)
            cmd_stats = self._get_command_stats(env_unwrapped)
            ee_stats = self._get_ee_stats(env_unwrapped)
            action_stats = self._get_action_stats(env_unwrapped)
            joint_stats = self._get_joint_limit_stats(env_unwrapped)
            parts = [f"step={step_count}"]
            parts.extend(f"{k}={float(v):.6f}" for k, v in log_data.items())
            parts.extend(cmd_stats)
            parts.extend(ee_stats)
            parts.extend(action_stats)
            parts.extend(joint_stats)
            self._fh.write(" ".join(parts) + "\n")
            self._fh.flush()
        return obs, rew, terminated, truncated, extras

    def _get_command_stats(self, env_unwrapped) -> list[str]:
        stats = []
        if not hasattr(env_unwrapped, "command_manager"):
            return stats
        left_name, right_name = self._command_names
        for name in (left_name, right_name):
            try:
                cmd = env_unwrapped.command_manager.get_command(name)
            except Exception:
                continue
            pos = cmd[:, :3].mean(dim=0)
            pos_std = cmd[:, :3].std(dim=0)
            quat = cmd[:, 3:7].mean(dim=0)
            stats.append(
                f"{name}_pos_mu=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
            )
            stats.append(
                f"{name}_pos_std=({pos_std[0]:.3f},{pos_std[1]:.3f},{pos_std[2]:.3f})"
            )
            stats.append(
                f"{name}_quat_mu=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})"
            )
        return stats

    def _get_ee_stats(self, env_unwrapped) -> list[str]:
        stats = []
        if not hasattr(env_unwrapped, "scene") or not hasattr(env_unwrapped, "command_manager"):
            return stats
        robot = env_unwrapped.scene["robot"]
        body_names = robot.data.body_names
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        left_cmd, right_cmd = self._command_names
        left_eef, right_eef = self._eef_names
        # Collect per-side stats, then add left/right symmetry deltas if both are available.
        side_cache = {}
        for eef_name, cmd_name, side in ((left_eef, left_cmd, "left"), (right_eef, right_cmd, "right")):
            try:
                eef_idx = body_names.index(eef_name)
            except ValueError:
                continue
            ee_pos_w = robot.data.body_pos_w[:, eef_idx]
            ee_quat_w = robot.data.body_quat_w[:, eef_idx]
            try:
                cmd = env_unwrapped.command_manager.get_command(cmd_name)
            except Exception:
                continue
            des_pos_b = cmd[:, :3]
            des_quat_b = cmd[:, 3:7]
            des_pos_w, des_quat_w = combine_frame_transforms(root_pos_w, root_quat_w, des_pos_b, des_quat_b)
            pos_err = torch.norm(ee_pos_w - des_pos_w, dim=1)
            ori_err = quat_error_magnitude(ee_quat_w, des_quat_w)
            ee_pos_mu = ee_pos_w.mean(dim=0)
            des_pos_mu = des_pos_w.mean(dim=0)
            stats.append(f"{side}_pos_err_mu={pos_err.mean():.6f}")
            stats.append(f"{side}_pos_err_std={pos_err.std():.6f}")
            stats.append(f"{side}_ori_err_mu={ori_err.mean():.6f}")
            stats.append(f"{side}_ori_err_std={ori_err.std():.6f}")
            stats.append(
                f"{side}_success_rate={(pos_err < self._pos_success_thresh).float().mean():.3f}"
            )
            stats.append(
                f"{side}_ori_success_rate={(ori_err < self._ori_success_thresh).float().mean():.3f}"
            )
            stats.append(
                f"{side}_ee_pos_mu=({ee_pos_mu[0]:.3f},{ee_pos_mu[1]:.3f},{ee_pos_mu[2]:.3f})"
            )
            stats.append(
                f"{side}_cmd_pos_mu=({des_pos_mu[0]:.3f},{des_pos_mu[1]:.3f},{des_pos_mu[2]:.3f})"
            )
            stats.append(
                f"{side}_ee_pos_0=({ee_pos_w[0,0]:.3f},{ee_pos_w[0,1]:.3f},{ee_pos_w[0,2]:.3f})"
            )
            stats.append(
                f"{side}_ee_quat_0=({ee_quat_w[0,0]:.3f},{ee_quat_w[0,1]:.3f},{ee_quat_w[0,2]:.3f},{ee_quat_w[0,3]:.3f})"
            )
            stats.append(
                f"{side}_cmd_pos_0=({des_pos_w[0,0]:.3f},{des_pos_w[0,1]:.3f},{des_pos_w[0,2]:.3f})"
            )
            stats.append(
                f"{side}_cmd_quat_0=({des_quat_w[0,0]:.3f},{des_quat_w[0,1]:.3f},{des_quat_w[0,2]:.3f},{des_quat_w[0,3]:.3f})"
            )
            side_cache[side] = {
                "ee_pos_w": ee_pos_w,
                "cmd_pos_w": des_pos_w,
            }

        # Symmetry diagnostics (x/z should match, y should be opposite).
        if "left" in side_cache and "right" in side_cache:
            l_ee = side_cache["left"]["ee_pos_w"]
            r_ee = side_cache["right"]["ee_pos_w"]
            l_cmd = side_cache["left"]["cmd_pos_w"]
            r_cmd = side_cache["right"]["cmd_pos_w"]
            ee_dx = (l_ee[:, 0] - r_ee[:, 0]).mean()
            ee_dy = (l_ee[:, 1] + r_ee[:, 1]).mean()
            ee_dz = (l_ee[:, 2] - r_ee[:, 2]).mean()
            cmd_dx = (l_cmd[:, 0] - r_cmd[:, 0]).mean()
            cmd_dy = (l_cmd[:, 1] + r_cmd[:, 1]).mean()
            cmd_dz = (l_cmd[:, 2] - r_cmd[:, 2]).mean()
            stats.append(f"ee_sym_d(x,y,z)_mu=({ee_dx:.3f},{ee_dy:.3f},{ee_dz:.3f})")
            stats.append(f"cmd_sym_d(x,y,z)_mu=({cmd_dx:.3f},{cmd_dy:.3f},{cmd_dz:.3f})")
        return stats

    def _get_action_stats(self, env_unwrapped) -> list[str]:
        stats = []
        if not hasattr(env_unwrapped, "action_manager"):
            return stats
        for name, side in (("left_arm_action", "left"), ("right_arm_action", "right")):
            try:
                term = env_unwrapped.action_manager.get_term(name)
            except Exception:
                continue
            act = term.processed_actions
            mag = torch.norm(act, dim=1)
            stats.append(f"{side}_action_mag_mu={mag.mean():.6f}")
            stats.append(f"{side}_action_mag_std={mag.std():.6f}")
            # Raw/unscaled actions if available (debug for sudden spikes)
            for attr, label in (("raw_actions", "raw"), ("actions", "in"), ("unscaled_actions", "unscaled")):
                if hasattr(term, attr):
                    raw = getattr(term, attr)
                    if raw is not None:
                        raw_mag = torch.norm(raw, dim=1)
                        stats.append(f"{side}_action_{label}_mag_mu={raw_mag.mean():.6f}")
                        stats.append(f"{side}_action_{label}_mag_std={raw_mag.std():.6f}")
                    break
        return stats

    def _get_joint_limit_stats(self, env_unwrapped) -> list[str]:
        stats = []
        if not hasattr(env_unwrapped, "action_manager"):
            return stats
        robot = env_unwrapped.scene["robot"]
        for name, side in (("left_arm_action", "left"), ("right_arm_action", "right")):
            try:
                term = env_unwrapped.action_manager.get_term(name)
            except Exception:
                continue
            joint_ids = term._joint_ids
            q = robot.data.joint_pos[:, joint_ids]
            limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
            q_min = limits[..., 0]
            q_max = limits[..., 1]
            # Log mean joint limits to verify left/right symmetry.
            q_min_mu = q_min.mean(dim=0)
            q_max_mu = q_max.mean(dim=0)
            span = torch.clamp(q_max - q_min, min=1.0e-6)
            margin = torch.minimum((q - q_min) / span, (q_max - q) / span)
            min_margin = margin.min(dim=1).values
            near = (min_margin < self._limit_margin_threshold).float()
            stats.append(f"{side}_limit_margin_min_mu={min_margin.mean():.6f}")
            stats.append(f"{side}_limit_near_rate={near.mean():.3f}")

            # per-joint stats
            joint_margin_mu = margin.mean(dim=0)
            joint_near_rate = (margin < self._limit_margin_threshold).float().mean(dim=0)
            if hasattr(term, "_joint_names"):
                names = term._joint_names
            else:
                names = [f"j{i}" for i in range(joint_margin_mu.numel())]
            margin_items = ",".join(
                f"{n}:{joint_margin_mu[i].item():.3f}" for i, n in enumerate(names)
            )
            near_items = ",".join(
                f"{n}:{joint_near_rate[i].item():.2f}" for i, n in enumerate(names)
            )
            min_items = ",".join(
                f"{n}:{q_min_mu[i].item():.3f}" for i, n in enumerate(names)
            )
            max_items = ",".join(
                f"{n}:{q_max_mu[i].item():.3f}" for i, n in enumerate(names)
            )
            stats.append(f"{side}_limit_margin_mu_per_joint=({margin_items})")
            stats.append(f"{side}_limit_near_rate_per_joint=({near_items})")
            stats.append(f"{side}_limit_min_mu_per_joint=({min_items})")
            stats.append(f"{side}_limit_max_mu_per_joint=({max_items})")
            # raw joint positions for env 0 (debug)
            q0 = q[0]
            if hasattr(term, "_joint_names"):
                names = term._joint_names
            else:
                names = [f"j{i}" for i in range(q0.numel())]
            q0_items = ",".join(f"{n}:{q0[i].item():.3f}" for i, n in enumerate(names))
            stats.append(f"{side}_joint_pos_0=({q0_items})")
            # Joint velocity stats
            if hasattr(robot.data, "joint_vel"):
                qd = robot.data.joint_vel[:, joint_ids]
                qd_mag = torch.norm(qd, dim=1)
                stats.append(f"{side}_joint_vel_mag_mu={qd_mag.mean():.6f}")
                stats.append(f"{side}_joint_vel_mag_std={qd_mag.std():.6f}")
            # Default pose distance (null-space attractor diagnostic)
            if hasattr(robot.data, "default_joint_pos"):
                q_ref = robot.data.default_joint_pos[:, joint_ids]
                dq = q - q_ref
                dq_abs = torch.abs(dq)
                stats.append(f"{side}_joint_def_abs_mu={dq_abs.mean():.6f}")
                stats.append(f"{side}_joint_def_abs_std={dq_abs.std():.6f}")
                dq0 = dq[0]
                dq0_items = ",".join(f"{n}:{dq0[i].item():.3f}" for i, n in enumerate(names))
                stats.append(f"{side}_joint_def_delta_0=({dq0_items})")
        return stats

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

EARLY_STOP_PATIENCE_ITERS = 1000
EARLY_STOP_MIN_DELTA = 1.0


class EarlyStopError(RuntimeError):
    """Raised when early stopping criteria are met."""

    def __init__(self, message: str, iteration: int, best_reward: float, best_iter: int):
        super().__init__(message)
        self.iteration = iteration
        self.best_reward = best_reward
        self.best_iter = best_iter


def _save_best_checkpoint(log_dir: str) -> None:
    """Save best checkpoint by Train/mean_reward to model_best.pt."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"[WARN] TensorBoard not available; best checkpoint not saved: {exc}")
        return

    tag = "Train/mean_reward"
    try:
        acc = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        acc.Reload()
    except Exception as exc:
        print(f"[WARN] Failed to read TensorBoard logs: {exc}")
        return

    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        print(f"[WARN] Tag '{tag}' not found; best checkpoint not saved.")
        return

    scalars = acc.Scalars(tag)
    if not scalars:
        print(f"[WARN] No scalar data for '{tag}'; best checkpoint not saved.")
        return

    best = max(scalars, key=lambda e: e.value)
    models = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    steps = []
    for name in models:
        try:
            step = int(name.split("_")[1].split(".")[0])
        except ValueError:
            continue
        steps.append(step)
    if not steps:
        print("[WARN] No model_*.pt checkpoints found; best checkpoint not saved.")
        return

    closest = min(steps, key=lambda s: abs(s - best.step))
    src = os.path.join(log_dir, f"model_{closest}.pt")
    dst = os.path.join(log_dir, "model_best.pt")
    shutil.copy2(src, dst)
    print(
        f"[INFO] Best checkpoint saved: {dst} (tag={tag}, best_step={best.step}, "
        f"best_value={best.value:.4f}, closest_step={closest})"
    )


def _attach_early_stop(runner: OnPolicyRunner | DistillationRunner, enabled: bool) -> None:
    """Wrap runner.log with early stopping on Train/mean_reward."""
    if not enabled:
        return
    if EARLY_STOP_PATIENCE_ITERS <= 0:
        return
    if getattr(runner, "disable_logs", False):
        return

    original_log = runner.log
    state: dict[str, Any] = {"best": None, "best_it": None}

    def _log_with_early_stop(locs: dict, width: int = 80, pad: int = 35):
        result = original_log(locs, width=width, pad=pad)
        rewbuffer = locs.get("rewbuffer", [])
        if not rewbuffer:
            return result
        mean_reward = sum(rewbuffer) / len(rewbuffer)
        it = int(locs.get("it", 0))

        if state["best"] is None or mean_reward > (state["best"] + EARLY_STOP_MIN_DELTA):
            state["best"] = float(mean_reward)
            state["best_it"] = it
        if state["best_it"] is not None and (it - state["best_it"]) >= EARLY_STOP_PATIENCE_ITERS:
            raise EarlyStopError(
                f"Early stop: no improvement > {EARLY_STOP_MIN_DELTA} for {EARLY_STOP_PATIENCE_ITERS} iterations.",
                iteration=it,
                best_reward=state["best"],
                best_iter=state["best_it"],
            )
        return result

    runner.log = _log_with_early_stop


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # LOG PATH RULE:
    #   <sbm_root>/log/rsl_rl/pipeline/<left|right|both>/<task_dir_name>/testN
    # side/folder are auto-resolved from task's env_cfg_entry_point (pipeline module path).
    task_name = args_cli.task
    side_dir, task_dir_name = _resolve_pipeline_log_components(task_name)
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_root_path = os.path.join(sbm_root, "log", "rsl_rl", "pipeline", side_dir, task_dir_name)
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # LOG RUN-NAME RULE:
    #   auto-increment "test1", "test2", ...
    # Change here if you want timestamp or custom run names.
    existing = []
    for name in os.listdir(log_root_path):
        if name.startswith("test"):
            suffix = name[4:]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = (max(existing) + 1) if existing else 1
    log_dir = f"test{next_idx}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # Environment-side logs/videos/checkpoints resolve from this run directory.
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # optional simple episode logging (reach_ik / grasp_2g_ik only)
    task_lower = args_cli.task.lower()
    if args_cli.log_able and (
        "reach_ik" in task_lower
        or "reachik" in task_lower
        or "grasp_2g_ik" in task_lower
        or "grasp2gik" in task_lower
        or "grasp_ik" in task_lower
        or "graspik" in task_lower
    ):
        log_name = f"{task_name}_{os.path.basename(log_dir)}.log"
        log_path = os.path.join(log_root_path, log_name)
        print(f"[INFO] Episode log: {log_path}")
        env = EpisodeLogWrapper(
            env,
            log_path,
            ("left_object_pose", "right_object_pose"),
            ("openarm_left_hand", "openarm_right_hand"),
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # log policy_low observation term order/dims for pouring (or any task that exposes policy_low)
    if hasattr(env.unwrapped, "observation_manager"):
        obs_mgr = env.unwrapped.observation_manager
        if "policy_low" in obs_mgr.active_terms:
            term_names = obs_mgr.active_terms["policy_low"]
            term_dims = obs_mgr.group_obs_term_dim["policy_low"]
            print("[INFO] policy_low obs term order/dims:")
            for name, dims in zip(term_names, term_dims):
                print(f"  - {name}: {tuple(dims)}")
            if obs_mgr.group_obs_concatenate.get("policy_low", False):
                print(f"[INFO] policy_low concatenated dim: {obs_mgr.group_obs_dim['policy_low']}")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        swap_lr=args_cli.swap_lr,
        swap_prob=args_cli.swap_lr_prob,
    )

    # optional warmstart rollout after each reset
    if args_cli.warmstart_ckpt and args_cli.warmstart_steps > 0:
        print(f"[INFO] Warmstart rollout enabled: ckpt={args_cli.warmstart_ckpt}, steps={args_cli.warmstart_steps}")
        if agent_cfg.class_name == "OnPolicyRunner":
            warm_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            warm_runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        warm_runner.load(args_cli.warmstart_ckpt)
        warm_policy = warm_runner.get_inference_policy(device=env.unwrapped.device)
        try:
            warm_policy_nn = warm_runner.alg.policy
        except AttributeError:
            warm_policy_nn = warm_runner.alg.actor_critic
        env = WarmStartWrapper(env, warm_policy, warm_policy_nn, args_cli.warmstart_steps)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    _attach_early_stop(runner, enabled=args_cli.able_early_stop)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO] Resume settings: load_run='{agent_cfg.load_run}', load_checkpoint='{agent_cfg.load_checkpoint}'")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    start_time = time.time()
    interrupted = False
    early_stopped = False
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        interrupted = True
        iteration = getattr(runner, "current_learning_iteration", "interrupt")
        interrupt_path = os.path.join(log_dir, f"model_interrupt_{iteration}.pt")
        print(f"[WARN] Training interrupted. Saving checkpoint to: {interrupt_path}")
        runner.save(interrupt_path)
    except EarlyStopError as exc:
        early_stopped = True
        early_path = os.path.join(log_dir, f"model_early_stop_{exc.iteration}.pt")
        print(
            f"[WARN] {exc} Saving checkpoint to: {early_path} "
            f"(best_reward={exc.best_reward:.4f} at iter {exc.best_iter})"
        )
        runner.save(early_path)
    finally:
        _save_best_checkpoint(log_dir)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    if interrupted:
        print("[INFO] Interrupted training checkpoint saved.")
    if early_stopped:
        print("[INFO] Early stopping checkpoint saved.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
