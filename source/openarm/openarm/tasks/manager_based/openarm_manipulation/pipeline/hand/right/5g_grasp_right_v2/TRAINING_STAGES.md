# 5g_grasp_right_v2 Staged Training

## Stage 1-3: Grasp-only pretraining (this task)

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
  --task "5g_grasp_right-v2" \
  --headless \
  --num_envs 2048
```

## Stage 4: Lift fine-tuning from grasp checkpoint

The lift task now exposes an additional agent key:
- `rl_games_finetune_from_5g_grasp_right_v2_cfg_entry_point`

Run:

```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
  --task "5g_lift_right-v1" \
  --agent rl_games_finetune_from_5g_grasp_right_v2_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  params.load_path='/ABS/PATH/TO/5g_grasp_right_v2/model.pth'
```

## Notes

- `params.load_path` must point to a valid rl_games checkpoint from `5g_grasp_right-v2`.
- If the checkpoint shape does not match because of policy architecture changes, keep the same network config between stages.
- For quick smoke tests, reduce env count (`--num_envs 64`) first.
