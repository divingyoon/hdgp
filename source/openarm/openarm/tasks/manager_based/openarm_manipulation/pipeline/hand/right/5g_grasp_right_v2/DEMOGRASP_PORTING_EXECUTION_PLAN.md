# 5g_grasp_right_v2 DemoGrasp Porting 실행계획

## 0. 목적
본 계획은 다음 3개 기능을 현재 `5g_grasp_right_v2` 환경에 단계적으로 반영하기 위한 실행 문서다.

1. `trackingReferenceFile` 스타일 시계열 참조 재생
2. object point-cloud feature 관측
3. contact sensor 기반 성공 판정/페널티

전제:
- 로봇: OpenArm right arm 7DOF + Tesollo right hand 20DOF
- 런타임: IsaacLab 2.3.1 + IsaacSim 5.1.0
- 대상 코드:
  - `/home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_grasp_right_v2/grasp_right_env.py`
  - `/home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_grasp_right_v2/grasp_right_env_cfg.py`

---

## 1. 전체 일정 (권장 순서)

### Phase A: 시계열 참조 재생
- 목표: 정책 액션 없이도 참조 궤적 기반 파지 동작 재생 가능
- 산출물: reference loader + replay 모드 + debug run 스크립트

### Phase B: contact 기반 성공/페널티
- 목표: 거리 기반 근사만 쓰던 성공조건을 접촉 기반으로 보강
- 산출물: contact sensor, success gate, table contact penalty

### Phase C: object PC feature 관측
- 목표: object identity/generalization을 위한 feature 관측 추가
- 산출물: object feature loader, obs 확장, 학습 설정 업데이트

### Phase D: 통합 검증
- 목표: A/B/C 기능 동시 활성화 시 학습/평가 정상 동작 검증
- 산출물: smoke 결과, 지표 체크리스트, 다음 튜닝 항목

---

## 2. Phase A 상세 계획: trackingReferenceFile 스타일 시계열 참조 재생

## 2.1 데이터 포맷 정의
- 파일 포맷(`.pkl`)
  - `wrist_initobj_pos`: `(T, 3)`
  - `wrist_quat`: `(T, 4)` (wxyz)
  - `hand_qpos`: `(T, 20)`
  - optional: `meta` (생성 로봇/좌표계/샘플링 주기)

## 2.2 설정 항목 추가 (`grasp_right_env_cfg.py`)
- `use_reference_replay: bool`
- `tracking_reference_file: str`
- `tracking_reference_lift_timestep: int`
- `randomize_tracking_reference: bool`
- `randomize_tracking_reference_range_xyzrpy: list[float]`
- `randomize_grasp_pose: bool`
- `randomize_grasp_pose_range: float`

## 2.3 런타임 로직 추가 (`grasp_right_env.py`)
- env init:
  - reference 파일 로드
  - tensor 캐싱 (`num_envs` 배치 확장)
- reset:
  - object-relative reference를 world로 변환
  - env별 랜덤 오프셋(위치/회전/hand pose)
- pre-step:
  - `progress_buf` 기반 timestep index 계산
  - 해당 시점의 wrist/hand target 사용
  - replay 모드면 policy action을 무시하거나 blending

## 2.4 검증 항목
- `debug_reference_replay` 실행 시
  - 손목/손가락 궤적이 프레임 점프 없이 재생
  - 에피소드마다 랜덤화 유효
  - out-of-range joint 명령 없음

## 2.5 완료 조건 (DoD)
- 참조 파일 1개로 재생 성공
- replay-only 모드에서 최소 100에피소드 연속 실행 에러 없음
- 추적 오차 로그(`eef pose`, `hand qpos`) 출력 가능

---

## 3. Phase B 상세 계획: contact sensor 기반 성공 판정/페널티

## 3.1 센서 구성 (`grasp_right_env_cfg.py`)
- 센서 대상: palm + fingertip body
- filter 대상 분리:
  - object contact
  - table contact

## 3.2 성공 판정 보강 (`grasp_right_env.py`)
- 기존 성공조건(거리 + hand error + hold steps)에 다음 gate 추가:
  - object contact force > threshold
  - k-step 연속 유지
- lift 성공은 object 높이 + contact 유지로 최종 판정

## 3.3 페널티 추가
- hand-table contact penalty
- object impact penalty (과도 충격)
- optional: self-collision penalty

## 3.4 검증 항목
- contact 로그 분리 확인: object/table
- 허위성공(False positive) 감소 확인
- 과도 밀기/넘어뜨리기 정책 감소

## 3.5 완료 조건 (DoD)
- success 판정에 contact gate가 실제 반영
- `grasp_success_contact_rate` 지표 로깅
- table contact가 reward에 음수로 반영

---

## 4. Phase C 상세 계획: object point-cloud feature 관측

## 4.1 데이터 준비
- object code -> feature 벡터 매핑 파일 준비 (`.pt` 또는 `.npy`)
- feature dim 고정(예: 64 또는 128)

## 4.2 설정 추가 (`grasp_right_env_cfg.py`)
- `use_object_pc_feat: bool`
- `object_pc_feat_path: str`
- `object_pc_feat_dim: int`

## 4.3 런타임 로직 (`grasp_right_env.py`)
- startup:
  - feature dict 로드
- reset:
  - active object에 해당하는 feature를 env buffer에 바인딩
- obs:
  - 기존 관측 끝에 feature concat
  - observation_space 자동 재계산

## 4.4 검증 항목
- object 변경 시 feature가 env별로 달라지는지 확인
- obs 차원과 네트워크 입력 차원 일치 확인

## 4.5 완료 조건 (DoD)
- 학습 시작/롤아웃 중 shape error 0건
- feature on/off ablation 가능

---

## 5. Phase D 상세 계획: 통합 검증

## 5.1 Smoke Test
- 256 env, 2k~5k steps 학습
- replay on/off 각각 1회
- contact on/off 각각 1회
- pc feature on/off 각각 1회

## 5.2 KPI
- success rate
- grasp_success_contact_rate
- table_contact_rate
- average lift height
- unstable termination rate (tipped/fallen)

## 5.3 완료 조건 (DoD)
- 기능 조합별 실행 안정성 확보
- 기준 실험 로그/체크포인트 산출
- 다음 튜닝 목록 도출

---

## 6. 파일 수정 계획 (예정)

## 신규 파일
- `.../5g_grasp_right_v2/reference_utils.py`
- `.../5g_grasp_right_v2/object_feature_utils.py`

## 수정 파일
- `.../5g_grasp_right_v2/grasp_right_env_cfg.py`
- `.../5g_grasp_right_v2/grasp_right_env.py`
- 필요 시 agent yaml (`config/agents/rl_games_ppo_cfg.yaml`)

---

## 7. 리스크 및 대응

1. 리스크: reference 좌표계 불일치
- 대응: object-local -> world 변환 유닛테스트 우선

2. 리스크: contact 임계값 과민/둔감
- 대응: force threshold sweep(3~5개) + 로그 비교

3. 리스크: obs 확장으로 학습 불안정
- 대응: pc feature off baseline 유지 후 점진 활성화

4. 리스크: 7+20 고자유도 탐색 난이도
- 대응: replay warm start + grasp-only 단계 우선

---

## 8. 실행 우선순위

1. Phase A (reference replay)
2. Phase B (contact success/penalty)
3. Phase C (object pc feature)
4. Phase D (통합 검증)

이 순서를 유지하면 디버깅 난이도와 실패 원인 분리가 가장 쉽다.
