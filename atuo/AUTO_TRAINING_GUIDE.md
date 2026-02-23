# SkillBlender 자동 학습 루프 가이드 (ATUO)

이 문서는 현재까지 구성한 자동 학습/평가/분석/재실험 루프의 사용법과 동작 방식을 정리합니다.

## 1) 목표
- IsaacLab + SkillBlender 환경에서 **실험 실행 → 평가 → 실패 분석 → 재실험** 자동화
- 학습은 결정론적으로 돌리고, **LLM은 해석/제안만 담당**
- 모든 결과는 파일로 남겨 재현 가능하게 유지

---

## 2) 기본 구조
- 실험 설정: `atuo/config/experiment.json`
- 실행 엔트리: `atuo/orchestrator.py`
- 학습 러너: `atuo/runner.py`
- 평가 스크립트: `atuo/eval_grasp2g_metrics.py`
- 실패 분석 + LLM: `atuo/analyzer.py`, `atuo/llm.py`

결과 저장 위치:
- 학습 로그: `hdgp/log/rsl_rl/<task>/testN`
- 자동 루프 로그: `hdgp/atuo/runs/<run_id>/`
  - `report.json`
  - `analysis_prompt.txt`
  - `analysis_response.txt`
  - `overrides.json`

---

## 3) 실행 방법

### (A) Ollama 서버가 켜져 있어야 함
```
ollama serve
ollama list
```

### (B) 자동 루프 실행
```
python3 ~/rl_ws/hdgp/atuo/orchestrator.py \
  --config ~/rl_ws/hdgp/atuo/config/experiment.json
```

GPU 지정과 num_envs 오버라이드 예시:
```
python3 ~/rl_ws/hdgp/atuo/orchestrator.py \
  --config ~/rl_ws/hdgp/atuo/config/experiment.json \
  --GPU=1 \
  --num_envs=3000
```

추가 CLI 오버라이드 예시:
```
python3 ~/rl_ws/hdgp/atuo/orchestrator.py \
  --config ~/rl_ws/hdgp/atuo/config/experiment.json \
  --task grasp2g-v1 \
  --agent rsl_rl_dual_cfg_entry_point \
  --max_iterations 5000 \
  --resume_from test8 \
  --resume_checkpoint model_600.pt \
  --gui
```

**로그가 터미널에 실시간 표시됨** (`stream_logs=true` 설정)

---

## 4) 평가 기준 (grasp2g-v1)
현재 성공 판정 기준은 **"lift > 0.1"** 그리고 **"object_tracking_dist < 0.05"** 를 기반으로 합니다.

`eval_grasp2g_metrics.py`가 계산하는 주요 지표:
- `lift_success_left/right`
- `goal_dist_min_left/right_mean` (object tracking 거리 최소값 평균)
- `success_rate_left/right` (lift + tracking dist 조건 만족)

최종 성공 판정:
- `success_rate_min` 이상인지 확인
- `goal_dist_mean_max` 이하인지 확인

설정 위치:
`hdgp/atuo/config/experiment.json`

---

## 5) 1000 이터레이션 단위 확인 (조기 중단)

설정:
```
"run_policy": {
  "train_chunk_iterations": 1000,
  "max_total_iterations": 10000,
  "stop_on_collapse": true
}
```

동작:
- 학습을 **1000 iter 단위로 나눠 실행**
- 매 chunk 후 **붕괴 신호**(mean_reward 급락, entropy collapse) 확인
- 붕괴 시 즉시 중단
- 붕괴가 없으면 다음 1000 iter 이어서 실행

즉, **무조건 1000에서 멈추는 게 아니라 1000마다 상태를 확인하는 방식**

---

## 6) 실패 분석 + LLM 제안 흐름

1) 실패 시 룰 기반으로 문제 분류
2) LLM이 "원인 요약 + override 제안" 생성
3) 허용된 override만 다음 run에 적용

설정 위치:
- 룰 기반 템플릿: `atuo/config/failure_rules.json`
- 허용 override 목록: `experiment.json` → `override_policy.allowed_overrides`

LLM 설정:
```
"llm": {
  "enabled": true,
  "provider": "ollama",
  "model": "qwen2.5:14b",
  "api_base": "http://localhost:11434"
}
```

---

## 7) 로그 확인 방법

- 현재 run 확인:
```
ls -lt ~/rl_ws/hdgp/atuo/runs | head -n 3
```

- 실행 로그 실시간 확인:
```
tail -f ~/rl_ws/hdgp/atuo/runs/<run_id>/train.stdout.txt
```

- IsaacLab 로그 폴더 확인:
```
ls -lt ~/rl_ws/hdgp/log/rsl_rl/grasp2g | head -n 3
```

---

## 8) report.md 자동 생성

실험이 끝날 때마다 `report.md`가 자동 생성됩니다.

위치:
```
~/rl_ws/hdgp/atuo/runs/<run_id>/report.md
```

포함 항목:
- Summary (성공/실패, task/agent)
- Training 지표 (mean_reward, entropy, iteration)
- Evaluation 지표 (success_rate, lift, tracking 거리)
- Analysis 요약 (LLM 분석 결과)
- 적용된 override 목록

---

## 9) 체크포인트에서 재시작

원하면 특정 실험 결과에서 재시작 가능:
```
./isaaclab.sh -p .../train.py \
  --resume --load_run test8 --checkpoint model_600.pt
```

자동화에 넣으려면 config에 다음을 추가하면 됨:
```
"train": {
  "resume_from": "test8",
  "resume_checkpoint": "model_600.pt"
}
```

(필요 시 바로 코드 반영 가능)

---

## 9) 파일 위치 요약
- 자동 루프 코드: `~/rl_ws/hdgp/atuo/`
- 설정 파일: `~/rl_ws/hdgp/atuo/config/experiment.json`
- 실패 룰: `~/rl_ws/hdgp/atuo/config/failure_rules.json`
- 평가 스크립트: `~/rl_ws/hdgp/atuo/eval_grasp2g_metrics.py`

---

## 10) 다음 확장 아이디어
- task별 eval 스크립트 추가
- reward/obs 변경을 위한 override 템플릿 확장
- 결과 보고서 자동 생성
