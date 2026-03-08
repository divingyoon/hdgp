# DemoGrasp 프레임 매칭 운영 가이드 (OpenArm/5g_grasp_right_v2)

작성일: 2026-03-06  
대상 런타임: IsaacLab 2.3.1 + IsaacSim 5.1.0  
대상 태스크: `5g_grasp_right-v2`  
현재 학습 로그 예시: `/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v2/test6`

---

## 1. 이 문서의 목적

이 문서는 "DemoGrasp 방식"을 현재 OpenArm 프레임워크에 어떻게 매칭해서 운영할지 정리한 실행 가이드다.

핵심은 다음 3가지다.

1. DemoGrasp처럼 단계를 분리해서 학습한다.
2. 각 단계에서 성공 기준(KPI)로 다음 단계 진입 여부를 결정한다.
3. right hand에서 검증된 절차를 다른 hand/task로 복제한다.

---

## 2. DemoGrasp 프레임과 현재 구현 매칭

1. Demo trajectory 기반 reference replay  
- 현재 구현: `use_reference_replay`, `tracking_reference_file` (Phase A)

2. contact 기반 성공 판정/페널티  
- 현재 구현: tip/palm object/table gate + penalty (Phase B)

3. object feature 관측 확장  
- 현재 구현: object code -> feature map 로드/concat (Phase C)

4. 조합 안정성 검증  
- 현재 구현: 8-case smoke matrix (Phase D full 완료)

요약:
- "프레임워크/파이프라인"은 구축 완료 상태다.
- 이제 핵심은 "학습 반복 + KPI 기반 튜닝"이다.

---

## 3. 지금 당장 학습을 돌릴 때 규칙

1. 우선 cup-only로 고정  
- `env.enable_cup=True`, `env.enable_primitives=False`
- 이유: primitive 경로는 별도 이슈가 남아 있어 본 학습 기준으로 부적합

2. DemoGrasp-like agent를 명시  
- `--agent rl_games_demograsp_like_cfg_entry_point`

3. batch/minibatch 나눗셈 규칙 준수  
- 현재 demograsp-like 설정은 `horizon_length=1`, `minibatch_size=2048`
- 따라서 기본으로는 `num_envs=2048`이어야 안전
- `num_envs`를 줄이면 반드시 `agent.params.config.minibatch_size`를 함께 줄여야 함

---

## 4. 권장 학습 순서 (실전)

## Stage 0: 설정 검증
1. 실행 명령으로 2~5분 smoke 학습
2. crash/shape error/NaN 여부 확인
3. 로그 경로 생성 확인 (`testN`)

예시:
```bash
cd /home/user/rl_ws/IsaacLab
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
  --task 5g_grasp_right-v2 \
  --agent rl_games_demograsp_like_cfg_entry_point \
  --headless \
  --num_envs 1024 \
  env.enable_cup=True \
  env.enable_primitives=False \
  agent.params.config.minibatch_size=1024
```

## Stage 1: 본 학습 (grasp 안정화)
1. 최소 1차 러닝 구간을 정해 연속 학습
2. 체크포인트 저장 주기 확인
3. 중간 KPI를 주기적으로 기록

권장:
- 첫 러닝: 3k~10k epoch 구간
- 이후: best ckpt 기준으로 연장 학습

## Stage 2: KPI 기반 선택
주요 KPI:
- `final_success`
- `grasp_success_contact_rate`
- `tip_table_contact_rate`
- `table_contact_penalty`

판정:
1. 성공률 상승
2. 테이블 접촉률 하락 또는 유지
3. contact 성공률 유지/상승

## Stage 3: Ablation/조합 확인
1. reference on/off
2. contact gate on/off
3. object feature on/off

목적:
- 어떤 기능이 실제 성능 향상에 기여하는지 분리 확인

---

## 5. "빠르게 돈다"가 정상인 이유

demograsp-like 설정은 `horizon_length=1` 기반이라 iteration이 빠르게 돈다.  
즉 로그 증가 속도는 빠르지만, 그 자체가 학습 완료를 의미하지는 않는다.

확인 포인트:
1. iteration 증가
2. reward/KPI 추세
3. checkpoint 품질

---

## 6. 다른 hand/task로 복제하는 방법

아래 순서 그대로 복제하면 된다.

1. `preset/constants/env_cfg/env` 4개 파일 구조를 동일하게 분리
2. reference replay 파라미터 추가
3. contact gate/penalty 추가
4. object feature concat 추가
5. 8-case smoke로 조합 안정성 확인
6. demograsp-like agent로 본 학습

핵심:
- 코드는 "복사"가 아니라 "구조 복제 + 링크/조인트 이름만 교체" 방식으로 진행
- KPI/DoD는 동일 기준으로 통일

---

## 7. 현재 상태 기준 다음 액션

1. `test6` 학습을 기준선(run baseline)으로 유지
2. 동일 조건으로 추가 1~2개 run을 더 만들어 분산 확인
3. 가장 좋은 ckpt를 선택해 play/eval 검증
4. 그 설정을 "right hand 표준 recipe"로 고정
5. 다음 hand/task로 동일 절차 복제

---

## 8. 운영 DoD

아래를 만족하면 "DemoGrasp 프레임 매칭 완료 + 학습 운영 가능"으로 본다.

1. cup-only 기준 장시간 학습 안정 수행
2. 8-case smoke에서 shape/crash 0건
3. KPI 기반으로 best run 선택 가능
4. 동일 레시피를 다른 hand/task에 복제 가능

