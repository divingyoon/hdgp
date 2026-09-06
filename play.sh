#!/usr/bin/env bash
# RL 모델 렌더링/플레이 래퍼 (GUI 화면 뷰어)
#
# 사용법:
#   ./play.sh <task_id> [추가 인자...]
#
# 예시:
#   ./play.sh open-tesol_r_reach_v5 --use_last_checkpoint
#   ./play.sh open-tesol_r_reach_v5 --checkpoint log/rl_games/open-tesol/right/reach-v5/test8/nn/last_open-tesol_r_reach_v5_ep_2_rew__0.9048067_.pth
#   ./play.sh open-tesol_r_reach_v5 --num_envs 4 --use_last_checkpoint

set -euo pipefail

TASK="${1:?'Usage: ./play.sh <task_id> [args...]'}"
shift 1

# 태스크명 뒤에 -play 가 없으면 자동 부착
if [[ "$TASK" != *-play ]]; then
    PLAY_TASK="${TASK}-play"
else
    PLAY_TASK="$TASK"
fi

HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${ISAACLAB_ROOT:-}" ]; then
    if [ -d "/home/usr/rl_ws/IsaacLab" ]; then
        ISAACLAB_ROOT="/home/usr/rl_ws/IsaacLab"
    elif [ -d "/home/user/rl_ws/IsaacLab" ]; then
        ISAACLAB_ROOT="/home/user/rl_ws/IsaacLab"
    else
        ISAACLAB_ROOT="$(cd "${HDGP_ROOT}/.." && pwd)/IsaacLab"
    fi
fi

echo "============================================"
echo " RL 플레이 / GUI 렌더링 시작"
echo "  태스크: $PLAY_TASK"
echo "============================================"

# --num_envs 인자가 전달되지 않은 경우 기본값 1 적용 (단일 로봇 뷰)
EXTRA_ARGS=()
if [[ ! " $* " =~ " --num_envs " ]]; then
    EXTRA_ARGS+=(--num_envs 1)
fi

"${ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${HDGP_ROOT}/scripts/reinforcement_learning/rl_games/play.py" \
    --task "$PLAY_TASK" \
    "${EXTRA_ARGS[@]}" \
    "$@"

