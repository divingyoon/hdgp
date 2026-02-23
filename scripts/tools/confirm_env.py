"""
confirm_env.py - 환경의 action/observation space 및 left/right 출력 순서 확인 스크립트

Usage:
    ./isaaclab.sh -p ../hdgp/scripts/tools/confirm_env.py --task Grasp2g_IK-v0
"""

import argparse

from isaaclab.app import AppLauncher

# argparse 먼저 설정 (AppLauncher 전에)
parser = argparse.ArgumentParser(description="환경 Action/Observation Space 확인")
parser.add_argument("--task", type=str, default="Grasp2g_IK-v0", help="Task ID")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--steps", type=int, default=3, help="Number of test steps")
parser.add_argument("--split_index", type=int, default=None, help="DualHead split index to verify")

# AppLauncher 인자 추가
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# headless 기본값 설정
if args_cli.headless is None:
    args_cli.headless = True

# Isaac Sim 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Sim 초기화 후 import ---
import torch
import gymnasium as gym

# Isaac Lab 유틸리티
from isaaclab_tasks.utils import parse_env_cfg

# Isaac Lab 환경 등록
import openarm.tasks


def print_separator(title: str = ""):
    """구분선 출력"""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def analyze_action_space(env):
    """Action space 상세 분석"""
    print_separator("ACTION SPACE 분석")

    action_space = env.action_space
    print(f"Action Space Type: {type(action_space).__name__}")
    print(f"Action Space Shape: {action_space.shape}")
    print(f"Action Space Low: {action_space.low}")
    print(f"Action Space High: {action_space.high}")

    # ManagerBasedRLEnv의 action manager 접근
    if hasattr(env, 'unwrapped'):
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'action_manager'):
            action_manager = unwrapped_env.action_manager
            print_separator("ACTION MANAGER 상세 정보")

            print(f"Total Action Dimension: {action_manager.total_action_dim}")
            print(f"Number of Action Terms: {len(action_manager.active_terms)}")
            print(f"\nAction Terms (순서대로):")

            cumulative_idx = 0
            for i, term_name in enumerate(action_manager.active_terms):
                term = action_manager._terms[term_name]
                term_dim = term.action_dim
                print(f"  [{i}] {term_name}:")
                print(f"      - Dimension: {term_dim}")
                print(f"      - Index Range: [{cumulative_idx} : {cumulative_idx + term_dim}]")

                # 추가 정보 출력
                if hasattr(term, 'joint_names'):
                    print(f"      - Joint Names: {term.joint_names}")
                if hasattr(term, '_body_idx'):
                    print(f"      - Body Index: {term._body_idx}")

                cumulative_idx += term_dim

            print(f"\n총 Action Dimension: {cumulative_idx}")
            return action_manager
    return None


def analyze_observation_space(env):
    """Observation space 상세 분석"""
    print_separator("OBSERVATION SPACE 분석")

    obs_space = env.observation_space
    print(f"Observation Space Type: {type(obs_space).__name__}")

    if hasattr(obs_space, 'spaces'):
        # Dict observation space
        for key, space in obs_space.spaces.items():
            print(f"\n[{key}]:")
            print(f"  Shape: {space.shape}")
            print(f"  Dtype: {space.dtype}")
    else:
        print(f"Shape: {obs_space.shape}")
        print(f"Dtype: {obs_space.dtype}")

    # Observation manager 접근
    if hasattr(env, 'unwrapped'):
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'observation_manager'):
            obs_manager = unwrapped_env.observation_manager
            print_separator("OBSERVATION MANAGER 상세 정보")

            import numpy as np
            for group_name in obs_manager._group_obs_term_names:
                print(f"\n[Group: {group_name}]")
                term_names = obs_manager._group_obs_term_names[group_name]
                term_dims = obs_manager._group_obs_term_dim[group_name]

                cumulative_idx = 0
                for term_name, term_dim in zip(term_names, term_dims):
                    dim_size = int(np.prod(term_dim))
                    print(f"  - {term_name}: shape={term_dim}, size={dim_size}, idx=[{cumulative_idx}:{cumulative_idx + dim_size}]")
                    cumulative_idx += dim_size
                print(f"  Total: {cumulative_idx}")


def test_action_output_order(env, action_manager, num_steps: int = 5):
    """실제 action 출력 순서 테스트"""
    print_separator("ACTION 출력 순서 테스트")

    # 환경 리셋
    obs, info = env.reset()
    print(f"Environment reset complete. Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")

    total_dim = action_manager.total_action_dim
    term_names = action_manager.active_terms

    # 각 action term별로 구분된 테스트 action 생성
    print(f"\n테스트: 각 action term에 고유 값 할당하여 순서 확인")

    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # 테스트용 action 생성: 각 term에 다른 값 할당
        action = torch.zeros(env.unwrapped.num_envs, total_dim, device=env.unwrapped.device)

        cumulative_idx = 0
        for i, term_name in enumerate(term_names):
            term = action_manager._terms[term_name]
            term_dim = term.action_dim

            # 각 term에 고유한 패턴의 값 할당
            # left: 양수, right: 음수로 구분
            if 'left' in term_name.lower():
                test_value = 0.1 * (i + 1)  # 양수
            elif 'right' in term_name.lower():
                test_value = -0.1 * (i + 1)  # 음수
            else:
                test_value = 0.0

            action[:, cumulative_idx:cumulative_idx + term_dim] = test_value
            cumulative_idx += term_dim

        print(f"입력 Action (env 0): {action[0].tolist()}")

        # Action 분해하여 각 term별 값 출력
        cumulative_idx = 0
        for term_name in term_names:
            term = action_manager._terms[term_name]
            term_dim = term.action_dim
            term_action = action[0, cumulative_idx:cumulative_idx + term_dim]
            print(f"  {term_name}: {term_action.tolist()}")
            cumulative_idx += term_dim

        # Step 실행
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward[0].item():.4f}")


def verify_dualhead_split(env, action_manager, split_index: int = None):
    """DualHead 정책의 split index 검증"""
    print_separator("DUALHEAD SPLIT INDEX 검증")

    term_names = action_manager.active_terms

    print(f"Action Terms 순서:")
    cumulative_idx = 0
    left_dims = 0
    right_dims = 0

    for term_name in term_names:
        term = action_manager._terms[term_name]
        term_dim = term.action_dim

        is_left = 'left' in term_name.lower()
        is_right = 'right' in term_name.lower()

        side = "LEFT" if is_left else ("RIGHT" if is_right else "OTHER")
        print(f"  [{cumulative_idx:2d}-{cumulative_idx + term_dim - 1:2d}] {term_name} ({side}, dim={term_dim})")

        if is_left:
            left_dims += term_dim
        elif is_right:
            right_dims += term_dim

        cumulative_idx += term_dim

    print(f"\n요약:")
    print(f"  Left actions total dimension: {left_dims}")
    print(f"  Right actions total dimension: {right_dims}")
    print(f"  Total dimension: {cumulative_idx}")

    # DualHead split index 권장값 계산
    recommended_split = left_dims
    print(f"\n권장 dof_split_index: {recommended_split}")
    print(f"  - Index 0 ~ {recommended_split - 1}: Left arm/hand actions")
    print(f"  - Index {recommended_split} ~ {cumulative_idx - 1}: Right arm/hand actions")

    if split_index is not None:
        if split_index == recommended_split:
            print(f"\n[OK] 현재 설정된 split_index ({split_index})가 올바릅니다!")
        else:
            print(f"\n[WARN] 현재 설정된 split_index ({split_index})와 권장값 ({recommended_split})이 다릅니다!")


def main():
    print_separator(f"환경 분석: {args_cli.task}")

    # 환경 설정 파싱
    device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
    )

    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"Environment created: {args_cli.task}")
    print(f"Number of envs: {env.unwrapped.num_envs}")

    # Action space 분석
    action_manager = analyze_action_space(env)

    # Observation space 분석
    analyze_observation_space(env)

    # DualHead split 검증
    if action_manager:
        verify_dualhead_split(env, action_manager, args_cli.split_index)

    # Action 출력 순서 테스트 (시뮬레이션 실행)
    if action_manager and args_cli.steps > 0:
        test_action_output_order(env, action_manager, args_cli.steps)

    print_separator("분석 완료")

    # 환경 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
