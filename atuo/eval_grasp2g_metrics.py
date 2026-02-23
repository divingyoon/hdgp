#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Evaluate grasp2g policy and write metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate grasp2g policy success metrics.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--goal_threshold", type=float, default=0.8)
    parser.add_argument("--lift_threshold", type=float, default=0.1)
    parser.add_argument("--goal_dist_threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


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


def main() -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    workspace_root = Path(__file__).resolve().parents[1]
    sbm_root = workspace_root / "hdgp"
    sys.path.append(str(sbm_root / "source"))

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import combine_frame_transforms
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import openarm.tasks  # noqa: F401
    from openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g import mdp as grasp_mdp

    results = {}

    def _goal_distance(env, command_name: str, object_name: str) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(
            env.scene["robot"].data.root_pos_w,
            env.scene["robot"].data.root_quat_w,
            des_pos_b,
        )
        obj_pos = env.scene[object_name].data.root_pos_w
        return torch.norm(des_pos_w - obj_pos, dim=1)

    @hydra_task_config(args.task, args.agent)
    def _run(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
        env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        rewards_cfg = env.unwrapped.cfg.rewards
        lift_height = max(float(args.lift_threshold), float(rewards_cfg.left_lifting_object.params.get("lift_height", 0.1)))
        goal_params = rewards_cfg.left_object_goal_tracking.params
        hold_duration = goal_params["phase_params"].get("hold_duration", 2.0)
        minimal_height = goal_params.get("minimal_height", 0.04)

        scene_keys = set(env.unwrapped.scene.keys())
        left_obj_name = "object" if "object" in scene_keys else "cup"
        right_obj_name = "object2" if "object2" in scene_keys else "cup2"
        left_obj_cfg = SceneEntityCfg(left_obj_name)
        right_obj_cfg = SceneEntityCfg(right_obj_name)
        left_ee_cfg = SceneEntityCfg("left_ee_frame")
        right_ee_cfg = SceneEntityCfg("right_ee_frame")

        num_envs = env.unwrapped.num_envs
        episode_count = 0
        lift_left = []
        lift_right = []
        hold_left = []
        hold_right = []
        goal_left = []
        goal_right = []
        left_goal_min_list = []
        right_goal_min_list = []
        left_track_success = []
        right_track_success = []
        success_left = []
        success_right = []

        lift_flag_l = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
        lift_flag_r = torch.zeros_like(lift_flag_l)
        hold_flag_l = torch.zeros_like(lift_flag_l)
        hold_flag_r = torch.zeros_like(lift_flag_l)
        goal_flag_l = torch.zeros_like(lift_flag_l)
        goal_flag_r = torch.zeros_like(lift_flag_l)
        hold_time_l = torch.zeros(num_envs, device=env.unwrapped.device)
        hold_time_r = torch.zeros_like(hold_time_l)
        left_goal_min = torch.full((num_envs,), float("inf"), device=env.unwrapped.device)
        right_goal_min = torch.full_like(left_goal_min, float("inf"))

        obs = env.get_observations()
        while episode_count < args.num_episodes:
            with torch.inference_mode():
                actions = _unwrap_actions(policy(obs))
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                lift_l = grasp_mdp.object_is_lifted(env.unwrapped, lift_height, left_obj_cfg) > 0.5
                lift_r = grasp_mdp.object_is_lifted(env.unwrapped, lift_height, right_obj_cfg) > 0.5

                goal_dist_l = _goal_distance(env.unwrapped, goal_params["command_name"], left_obj_cfg.name)
                goal_dist_r = _goal_distance(
                    env.unwrapped,
                    rewards_cfg.right_object_goal_tracking.params["command_name"],
                    right_obj_cfg.name,
                )

                goal_l = (
                    grasp_mdp.object_goal_distance_with_ee(
                        env.unwrapped,
                        std=goal_params["std"],
                        minimal_height=minimal_height,
                        command_name=goal_params["command_name"],
                        object_cfg=left_obj_cfg,
                        ee_frame_cfg=left_ee_cfg,
                        reach_std=goal_params["reach_std"],
                    )
                    > args.goal_threshold
                )
                goal_r = (
                    grasp_mdp.object_goal_distance_with_ee(
                        env.unwrapped,
                        std=goal_params["std"],
                        minimal_height=minimal_height,
                        command_name=rewards_cfg.right_object_goal_tracking.params["command_name"],
                        object_cfg=right_obj_cfg,
                        ee_frame_cfg=right_ee_cfg,
                        reach_std=goal_params["reach_std"],
                    )
                    > args.goal_threshold
                )

            lift_flag_l |= lift_l
            lift_flag_r |= lift_r

            hold_time_l = torch.where(lift_l, hold_time_l + env.unwrapped.step_dt, torch.zeros_like(hold_time_l))
            hold_time_r = torch.where(lift_r, hold_time_r + env.unwrapped.step_dt, torch.zeros_like(hold_time_r))
            hold_flag_l |= hold_time_l >= hold_duration
            hold_flag_r |= hold_time_r >= hold_duration

            goal_flag_l |= goal_l
            goal_flag_r |= goal_r
            left_goal_min = torch.minimum(left_goal_min, goal_dist_l)
            right_goal_min = torch.minimum(right_goal_min, goal_dist_r)

            if torch.any(dones):
                done_ids = torch.nonzero(dones).squeeze(-1)
                for idx in done_ids.tolist():
                    if episode_count >= args.num_episodes:
                        break
                    lift_left.append(bool(lift_flag_l[idx].item()))
                    lift_right.append(bool(lift_flag_r[idx].item()))
                    hold_left.append(bool(hold_flag_l[idx].item()))
                    hold_right.append(bool(hold_flag_r[idx].item()))
                    goal_left.append(bool(goal_flag_l[idx].item()))
                    goal_right.append(bool(goal_flag_r[idx].item()))
                    left_goal_dist = float(left_goal_min[idx].item())
                    right_goal_dist = float(right_goal_min[idx].item())
                    left_goal_min_list.append(left_goal_dist)
                    right_goal_min_list.append(right_goal_dist)
                    left_track_success.append(left_goal_dist <= args.goal_dist_threshold)
                    right_track_success.append(right_goal_dist <= args.goal_dist_threshold)
                    success_left.append(bool(lift_flag_l[idx].item()) and left_goal_dist <= args.goal_dist_threshold)
                    success_right.append(bool(lift_flag_r[idx].item()) and right_goal_dist <= args.goal_dist_threshold)
                    episode_count += 1

                lift_flag_l[done_ids] = False
                lift_flag_r[done_ids] = False
                hold_flag_l[done_ids] = False
                hold_flag_r[done_ids] = False
                goal_flag_l[done_ids] = False
                goal_flag_r[done_ids] = False
                hold_time_l[done_ids] = 0.0
                hold_time_r[done_ids] = 0.0
                left_goal_min[done_ids] = float("inf")
                right_goal_min[done_ids] = float("inf")

        def _mean(x):
            return sum(x) / len(x) if x else 0.0

        results.update(
            {
                "episodes": args.num_episodes,
                "lift_success_left": _mean(lift_left),
                "lift_success_right": _mean(lift_right),
                "hold_success_left": _mean(hold_left),
                "hold_success_right": _mean(hold_right),
                "goal_success_left": _mean(goal_left),
                "goal_success_right": _mean(goal_right),
                "goal_dist_min_left_mean": _mean(left_goal_min_list),
                "goal_dist_min_right_mean": _mean(right_goal_min_list),
                "goal_track_success_left": _mean(left_track_success),
                "goal_track_success_right": _mean(right_track_success),
                "success_rate_left": _mean(success_left),
                "success_rate_right": _mean(success_right),
                "goal_threshold": args.goal_threshold,
                "goal_dist_threshold": args.goal_dist_threshold,
                "lift_height": float(lift_height),
                "lift_threshold": float(args.lift_threshold),
                "hold_duration": float(hold_duration),
                "minimal_height": float(minimal_height),
            }
        )

        env.close()

    _run()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if hasattr(simulation_app, "close"):
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
