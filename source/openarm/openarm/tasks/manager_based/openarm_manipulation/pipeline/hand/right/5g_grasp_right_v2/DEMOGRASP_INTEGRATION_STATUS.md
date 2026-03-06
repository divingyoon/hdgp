# DemoGrasp 연계 현황 (5g_grasp_right_v2 기준)

작성일: 2026-03-06  
대상 코드베이스:
- `/home/user/rl_ws/hdgp`
- 참고: `/home/user/rl_ws/repo/DemoGrasp`, `/home/user/rl_ws/repo/DemoGrasp_ilab221`

---

## 1. 목표와 기본 원칙

이번 작업의 목표는 `DemoGrasp`를 통째로 이식하는 것이 아니라, 아래 3가지를 현재 OpenArm right-hand 파이프라인에 흡수하는 것이었다.

1. hand abstraction (핸드/로봇 메타데이터 분리)
2. reference-conditioned pregrasp (object-relative pregrasp 레퍼런스)
3. grasp-only 우선 학습 구조 (grasp 단계 분리)

추가로 4단계(운영 연결)로, grasp-only 체크포인트를 `5g_lift_right-v1` 파인튜닝에 연결했다.

---

## 2. 현재 반영된 단계별 결과

## 2.1 1단계: 구조 분리 (hand abstraction)

### 반영 내용
- 신규 태스크 생성: `5g_grasp_right_v2`
- hand/robot 프리셋 모듈 추가로 하드코딩 분리

### 핵심 파일
- `grasp_right_preset.py`
- `grasp_right_constants.py`
- `grasp_right_env_cfg.py`
- `grasp_right_env.py`

### DemoGrasp 대응 개념
DemoGrasp의 `tasks/hand/*.yaml` 역할을, v2에서 `grasp_right_preset.py`로 재구성.

### 실제 분리된 항목
- 오른팔/오른손/왼팔 조인트 그룹
- 왼팔 고정(rest) 자세
- hand link 목록(USD/Fabrics)
- start/grasp pose, PCA min/max
- object spawn/goal 기본값
- palm workspace 계산 함수

---

## 2.2 2단계: pregrasp reference 도입

### 반영 내용
`DemoGrasp`의 reference-conditioned 철학을 object-relative pregrasp target으로 도입.

### 핵심 파일
- `grasp_right_env_cfg.py`
- `grasp_right_env.py`
- `grasp_right_constants.py`

### 추가된 설정 파라미터 (cfg)
- `pregrasp_offset_x/y/z`
- `pregrasp_noise_x/y/z`
- `pregrasp_activate_dist`
- `pregrasp_reach_weight`
- `pregrasp_reach_sharpness`

### 런타임 로직
- reset 시 object spawn 후 `pregrasp_pos = object_pos + offset + noise`
- 관측에 다음 추가:
  - `object_init_pos`
  - `pregrasp_delta = pregrasp_pos - palm_center`
- 보상에 `pregrasp_reward` 추가
- 단계 게이트 기준을 `palm_dist` 중심에서 `pregrasp_dist` 중심으로 전환

### 관측 차원 변경
- 150 -> 156

---

## 2.3 3단계: grasp-only 분리

### 반영 내용
lift/goal 중심 학습 전에 grasp 형성 자체를 안정화하도록 모드 분리.

### 핵심 파일
- `grasp_right_env_cfg.py`
- `grasp_right_env.py`

### 추가된 설정 파라미터
- `grasp_only_mode`
- `terminate_on_grasp_success`
- `grasp_success_hold_steps`
- `grasp_success_palm_dist`
- `grasp_success_hand_error`
- `grasp_success_max_height_delta`
- `grasp_stability_weight`
- `grasp_only_goal_reward_scale`
- `grasp_only_lift_reward_scale`

### 런타임 로직
- `grasp_formed` 판정 함수 추가
  - palm 거리 + hand grasp pose 오차 + 과도한 높이 변화 제한
- 연속 유지 스텝 카운트(`grasp_hold_steps`)로 성공 판정
- 성공 시 종료(옵션)
- grasp-only 모드에서 goal/lift 보상 기본 비활성

---

## 2.4 4단계: lift 파인튜닝 연결

### 반영 내용
`5g_grasp_right-v2` 체크포인트를 `5g_lift_right-v1`에서 직접 받아 파인튜닝할 수 있는 agent entry-point 추가.

### 핵심 파일
- `5g_lift_right_v1/config/agents/rl_games_ppo_finetune_from_5g_grasp_right_v2_cfg.yaml`
- `5g_lift_right_v1/config/__init__.py`
- `5g_grasp_right_v2/TRAINING_STAGES.md`

### 추가된 agent 키
- `rl_games_finetune_from_5g_grasp_right_v2_cfg_entry_point`

### 의도
- Stage 1~3(grasp-only) 학습 결과를 Stage 4(lift) 초기값으로 연결
- 학습 실패 원인 분리 및 수렴 안정성 개선

---

## 3. DemoGrasp와의 매핑 표

| DemoGrasp 개념 | 현재 v2 반영 상태 | 반영 위치 |
|---|---|---|
| hand-specific config(yaml) | 반영 완료(파이썬 preset) | `grasp_right_preset.py` |
| tracking reference 기반 접근 | 부분 반영(pregrasp 단일 레퍼런스) | `grasp_right_env_cfg.py`, `grasp_right_env.py` |
| randomizeTrackingReference/randomizeGraspPose | 부분 반영(pregrasp offset 노이즈) | reset 로직 |
| grasp-only 우선 학습 | 반영 완료(모드/성공 기준/보상 게이트) | `grasp_right_env_cfg.py`, `grasp_right_env.py` |
| grasp -> lift 단계 전환 | 반영 완료(학습 실행 경로 연결) | lift agent config + 등록 |
| single demonstration trajectory replay | 미반영 | (추후 필요 시) |
| object point cloud 중심 vision 파이프라인 | 미반영 | (현 단계 비포함) |

---

## 4. 현재 실행 경로

## 4.1 grasp-only 학습
```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
  --task "5g_grasp_right-v2" \
  --headless \
  --num_envs 2048
```

## 4.2 lift 파인튜닝 (grasp ckpt 로드)
```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
  --task "5g_lift_right-v1" \
  --agent rl_games_finetune_from_5g_grasp_right_v2_cfg_entry_point \
  --headless \
  --num_envs 2048 \
  params.load_path='/ABS/PATH/TO/5g_grasp_right_v2/model.pth'
```

---

## 5. 아직 미반영인 항목 (의도적으로 보류)

1. Demo trajectory 파일(`trackingReferenceFile`과 동등한 시계열 참조) 직접 재생
2. vision/object point cloud 기반 관측 파이프라인
3. pour 태스크 확장(stage 5)
4. object-relative orientation reference(현재는 위치 중심)

---

## 6. 변경 파일 목록 요약

## 신규 (v2)
- `.../5g_grasp_right_v2/grasp_right_preset.py`
- `.../5g_grasp_right_v2/grasp_right_constants.py`
- `.../5g_grasp_right_v2/grasp_right_env_cfg.py`
- `.../5g_grasp_right_v2/grasp_right_env.py`
- `.../5g_grasp_right_v2/grasp_right_utils.py`
- `.../5g_grasp_right_v2/config/__init__.py`
- `.../5g_grasp_right_v2/config/agents/rl_games_ppo_cfg.yaml`
- `.../5g_grasp_right_v2/TRAINING_STAGES.md`
- `.../5g_grasp_right_v2/DEMOGRASP_INTEGRATION_STATUS.md`

## 수정 (lift 연결)
- `.../5g_lift_right_v1/config/__init__.py`
- `.../5g_lift_right_v1/config/agents/rl_games_ppo_finetune_from_5g_grasp_right_v2_cfg.yaml` (신규 추가)

---

## 7. 결론

현재 상태에서 `DemoGrasp` 연계는 다음 수준까지 완료됐다.

- 구조: hand abstraction 반영 완료
- 초기 탐색: pregrasp reference + reset randomization 반영 완료
- 학습 절차: grasp-only 분리 완료
- 운영 연결: grasp ckpt -> lift fine-tune 경로 반영 완료

즉, "DemoGrasp를 복제"한 것이 아니라, 문서 원칙대로 "핵심 설계만 OpenArm fabrics 기반 태스크에 흡수"한 상태다.
