# ATUO: SkillBlender Experiment Loop (MVP)

This folder contains a minimal, deterministic experiment loop for hdgp + IsaacLab.
It follows: plan -> run -> eval -> decide -> report.

## Quick start
1) Edit config: ~/rl_ws/hdgp/atuo/config/experiment.json
2) Run orchestrator:
   python ~/rl_ws/hdgp/atuo/orchestrator.py --config ~/rl_ws/hdgp/atuo/config/experiment.json

## What it does
- Runs IsaacLab training via ./isaaclab.sh
- Finds the newest log directory under hdgp/log/rsl_rl/<task>
- Extracts key train metrics from TensorBoard events
- Runs evaluation for grasp2g-v1 and writes metrics.json
- Applies deterministic success criteria and produces a run report
- On failure, calls LLM (OpenAI) to suggest overrides and records analysis

## Files
- orchestrator.py: state machine loop
- runner.py: deterministic CLI runner for IsaacLab
- metrics.py: TensorBoard parsing + metrics.json assembly
- eval_grasp2g_metrics.py: evaluation script that writes metrics.json
- config/experiment.json: editable run config

## LLM setup
- OpenAI: set `OPENAI_API_KEY` and keep `llm.provider="openai"`.
- Local (Ollama): set `llm.provider="ollama"` and `llm.api_base="http://localhost:11434"`.
- Configure model/temperature in `~/rl_ws/hdgp/atuo/config/experiment.json`.

## Notes
- This MVP is task-specific for grasp2g-v1 evaluation. Other tasks should add a new eval script.
- The loop is deterministic as long as seeds and configs are fixed.
