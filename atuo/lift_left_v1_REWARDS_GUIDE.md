# 5G Lift Left v1 - 보상함수 상세 가이드 (LLM/Orchestrator용)

이 문서는 `orchestrator.py` + LLM(ollama)이 **보상 weight/params override를 안전하게 제안**할 수 있도록,
- 함수별 계산식
- 활성화 게이트(연결 흐름)
- weight 증감 시 기대 효과/부작용
을 최신 코드 기준으로 정리한 문서입니다.

기준 코드:
- `hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/lift_left_env_cfg.py`
- `hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v1/mdp/rewards.py`

---

## 1) 전체 연결 흐름

### 1-1. 하드 트리거 (DexPour)
- `λ (approach)`
  - `λ = 1[ ||p_ee - p_target_static|| < 0.05 ]`
- `μ (grasp)`
  - `μ = λ * 1[ n_contact >= 3 ]`
  - `n_contact`: fingertip geometry contact count
- `ν (lift)`
  - `ν = μ * 1[ z_cup >= z_init + 0.04 ]`

### 1-2. 보상 활성 흐름(요약)
1. Reach 유도: `reaching_object`, `reaching_object_fine`, `end_effector_orientation`
2. Reach 이후 hand closing: `thumb_grasp`, `pinky_grasp`, `synergy_grip`, `thumb_tip_z`, `synergy_tip_z`, `ee_descent`
3. Lift: `lifting_object`, `cup_lift_progress`
4. Goal tracking: `object_goal_tracking`, `object_goal_tracking_fine_grained`
5. 전구간 패널티/정규화: `object_displacement`, `finger_normal_range`, `action_rate`, `joint_vel`

중요:
- 현재 코드에서 `reaching_object`, `reaching_object_fine`, `end_effector_orientation`은 하드 `(1-λ)` 게이트가 없음(항상 계산).
- `ee_descent`는 `λ*(1-μ)`로 동작.

---

## 2) 핵심 파라미터

- `grasp2g_target_offset = (0.01, -0.06, 0.08)`
- dynamic reach z:
  - `reach_dynamic_z_high=0.25`
  - `reach_dynamic_xy_hi=0.10`
  - `reach_dynamic_xy_lo=0.03`
  - `reach_dynamic_xy_gate=0.03`
  - `reach_dynamic_z_descent_rate=0.001`
- reach 중 cup 밀림 억제:
  - `reach_displacement_free_threshold=0.015`
  - `reach_displacement_suppress_scale=0.03`
- displacement shaping:
  - `displacement_penalty_scale=0.02`
  - `displacement_penalty_power=2.0`
  - `displacement_penalty_gate_mix=0.5`

---

## 3) 보상 합산 수식

- raw term: `r_i`
- weighted term: `R_i = w_i * r_i`
- total step reward: `R_total = Σ_i R_i`

`tanh kernel` 표기:
- `K(d, s) = 1 - tanh(d/s)`

---

## 4) 항목별 상세 (수식 + 활성 게이트 + weight 증감)

### 4-1. Reaching Phase

#### `reaching_object` (`object_ee_distance`, weight `8.0`)
- 수식:
  - `r = K(||p_target_dyn - p_ee||, 0.15) * exp(-max(d_xy - d_free, 0)/s_disp)`
- 역할:
  - 멀리서 EE를 grasp target으로 끌어오는 주 gradient
- weight 증감:
  - 증가: 접근 속도↑, 대신 컵 밀기/충돌 행동 증가 가능
  - 감소: grasp 학습 전환은 빨라질 수 있으나 초기 접근 실패 증가

#### `reaching_object_fine` (`object_ee_distance_fine`, weight `10.0`)
- 수식:
  - `r = K(||p_target_static - p_ee||, 0.065)`
- 역할:
  - target 근처 정밀 접근
- weight 증감:
  - 증가: 최종 접근 정밀도↑, 과하면 손가락 닫기 전환이 늦어짐
  - 감소: 빠른 전환 가능, 대신 contact 전 EE 오차 커짐

#### `end_effector_orientation` (`eef_z_perpendicular_object_z`, weight `4.0`)
- 수식:
  - `cos = dot(z_ee, z_obj)`
  - `r = K(|cos|, 0.3)`
- 역할:
  - EE z축이 cup z축과 수직에 가까운 orientation 유도
- weight 증감:
  - 증가: 자세 안정↑, reach 속도↓ 가능
  - 감소: 빠른 접근 대신 grasp 자세 불량 가능

#### `thumb_reaching_pose` (weight `0.5`)
- 게이트: `(1-λ)`
- 수식:
  - `e = Σ(q_thumb - q_open)^2`
  - `r = (1-λ) * K(e, 1.0)`
- 증감:
  - 증가: 엄지 premature closing 방지↑
  - 감소: 조기 닫힘 가능

#### `pinky_reaching_pose` (weight `0.5`)
- 게이트: `(1-λ)`
- 수식:
  - `e = Σ(q_pinky - q_open)^2`
  - `r = (1-λ) * K(e, 1.0)`
- 증감:
  - 증가: pinky 조기 닫힘 억제
  - 감소: grasp 전 손가락 형상 안정성 저하 가능

#### `synergy_reaching_pose` (weight `0.5`)
- 게이트: `(1-λ)`
- 수식:
  - `e = Σ(q_synergy - 0)^2`
  - `r = (1-λ) * K(e, 5.0)`
- 증감:
  - 증가: reach 중 synergy 개방 유지
  - 감소: reach 중 synergy 조기 수축 가능

### 4-2. Grasp Phase

#### `thumb_grasp` (weight `15.0`)
- 게이트: `λ`
- 수식:
  - `rad = ||p_thumb_xy - p_cup_xy||`
  - `r_surface = K(|rad - r_cup|, 0.05)`
  - `r_pen = tanh(max(r_cup-rad,0)/0.01)`
  - `z_gate = 1 - tanh(max(z_thumb-z_f2,0)/0.03)`
  - `r = λ * (r_surface - r_pen) * z_gate`
- 증감:
  - 증가: 엄지 접촉 형성↑, 과하면 엄지 과집중
  - 감소: 엄지 참여 부족으로 3점 접촉 실패 가능

#### `pinky_grasp` (weight `12.0`)
- 게이트: `λ`
- 수식:
  - `rad = ||p_pinky_xy - p_cup_xy||`
  - `r = λ * (K(|rad-r_cup|, 0.05) - tanh(max(r_cup-rad,0)/0.01))`
- 증감:
  - 증가: pinky 지지력↑
  - 감소: 대칭 파지 약화

#### `synergy_grip` (weight `20.0`)
- 게이트: `λ`
- 수식:
  - `close = clamp((a_synergy+1)/2, 0,1)`
  - `g_surface = K(mean_k | ||p_fk_xy-p_cup_xy|| - r_cup |, 0.01), k={2,3,4}`
  - `r = λ * g_surface * close`
- 증감:
  - 증가: closing action 적극화, 과하면 허공 closing/진동 위험(표면게이트가 일부 완화)
  - 감소: grasp 형성 느림, lift 진입 지연

#### `thumb_tip_z` (weight `8.0`)
- 게이트: `λ`
- 수식:
  - `z_term = K(max(z_thumb-z_f2,0), 0.10)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/0.06)`
  - `r = λ * z_term * xy_gate`
- 증감:
  - 증가: 엄지 높이 정렬 강화
  - 감소: 엄지 위로 뜨는 failure 증가 가능

#### `synergy_tip_z` (weight `8.0`)
- 게이트: `λ`
- 수식:
  - `z_term = K(|z_f2_tip-(z_cup+0.09)|, 0.06)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/0.06)`
  - `r = λ * z_term * xy_gate`
- 증감:
  - 증가: synergy finger 높이 정렬 강화
  - 감소: cup 상단 랩핑 품질 저하 가능

#### `ee_descent` (weight `10.0`)
- 게이트: `λ*(1-μ)`
- 수식:
  - `z_term = K(|z_ee-(z_cup+0.04)|, 0.04)`
  - `xy_gate = exp(-||p_ee_xy-p_cup_xy||/0.06)`
  - `r = λ*(1-μ)*z_term*xy_gate`
- 증감:
  - 증가: grasp 직전 z 접근 강화
  - 과증가: 과도한 하강으로 cup disturbance 가능

### 4-3. Lift / Goal Phase

#### `lifting_object` (weight `10.0`)
- 게이트: `μ`
- 수식:
  - `r = μ * 1[z_cup > z_init + 0.04]`
- 증감:
  - 증가: binary success 신호 강조
  - 감소: lift 성공 강화 약화

#### `cup_lift_progress` (weight `20.0`)
- 게이트: `μ`
- 수식:
  - `r = μ * tanh(max(z_cup-z_init,0)/0.05)`
- 증감:
  - 증가: 들어올리는 연속 gradient 강화
  - 감소: lift 시작 지연

#### `object_goal_tracking` (weight `20.0`)
- 게이트: `ν`
- 수식:
  - `r = ν * K(||p_obj-p_goal||, 0.3)`
- 증감:
  - 증가: lift 후 목표 이동 강조
  - 감소: lift는 되는데 목표 추적 약화

#### `object_goal_tracking_fine_grained` (weight `10.0`)
- 게이트: `ν`
- 수식:
  - `r = ν * K(||p_obj-p_goal||, 0.1)`
- 증감:
  - 증가: goal 근처 미세 정렬 강화
  - 감소: 최종 정밀도 저하 가능

### 4-4. Penalty / Regularization

#### `object_displacement` (`weight=-5.0`)
- raw 수식:
  - `d = ||p_cup_xy-p_init_xy||`
  - `p = (max(d-0.01,0)/0.02)^2`
  - `p = p * ((1-0.5) + 0.5*g_grasp_progress) * (1-μ)`
  - `r_raw = p` (양수), 최종기여 `R = -5.0 * r_raw`
- weight(절댓값) 증감:
  - 절댓값 증가: cup 밀기 강하게 억제, 과하면 탐색 위축
  - 절댓값 감소: reach는 빨라질 수 있으나 cup pushing 증가

#### `finger_normal_range` (`weight=-2.0`)
- raw 수식:
  - `v = Σ_j [max(lo_j-q_j,0)+max(q_j-hi_j,0)]`
  - `R = -2.0 * v`
- 증감:
  - 절댓값 증가: 관절 안정↑, 과하면 grasp 자유도 제약
  - 절댓값 감소: 비정상 관절 자세 증가 가능

#### `action_rate` (`weight=-1e-4`)
- `R = -1e-4 * ||a_t-a_{t-1}||^2`

#### `joint_vel` (`weight=-1e-4`)
- `R = -1e-4 * ||qdot_left||^2`

---

## 5) 비활성(현재 weight=0) 항목

- `finger_tip_to_cup` (`finger_wrap_cylinder_reward`)
- `finger_wrap_coverage` (`finger_wrap_coverage_reward`)

필요 시 재활성화:
- `env.rewards.finger_tip_to_cup.weight`
- `env.rewards.finger_wrap_coverage.weight`

---

## 6) Hydra Override 가이드 (orchestrator 직접 사용)

### 6-1. weight 조정 키
- 형식: `env.rewards.<term>.weight=<value>`
- 예시:
  - `env.rewards.synergy_grip.weight=24.0`
  - `env.rewards.object_displacement.weight=-6.0`

### 6-2. params 조정 키
- 형식: `env.rewards.<term>.params.<param>=<value>`
- 예시:
  - `env.rewards.ee_descent.params.target_z_offset=0.035`
  - `env.rewards.thumb_grasp.params.std=0.04`

### 6-3. 권장 조정 스텝
- 큰 항목(10~20대 weight): 한 번에 `±10~20%`
- 중간 항목(1~10): 한 번에 `±10~30%`
- 패널티 항목: 절댓값 기준 `±10~20%`

---

## 7) Termination

- `cup_dropping`: `z_cup < -0.05`
- `cup_tipping`: `dot(z_cup_axis, z_world) < cos(90°)`

---

## 8) 빠른 진단 규칙 (LLM용)

- `synergy_grip` 낮고 `reaching_*` 높음:
  - `synergy_grip.weight` 소폭↑ 또는 `thumb/pinky_grasp` 소폭↑
- `object_displacement` 절댓값 급증:
  - `object_displacement` 절댓값↑, `reaching_object.weight` 소폭↓
- `lifting_object` 거의 0:
  - `cup_lift_progress.weight`↑, `ee_descent.weight` 미세↑
- lift는 되는데 goal tracking 저조:
  - `object_goal_tracking(_fine)` weight↑

---

## 9) `rl_games` 학습 에이전트 설명

### 9-1. 이 태스크에서 쓰는 기본 `rl_games` 에이전트
- task id `5g_lift_left-v1`는 `rl_games_cfg_entry_point`를 통해
  `config/agents/rl_games_ppo_cfg.yaml`을 사용.
- 학습 실행 시 `--agent rl_games_cfg_entry_point`를 주면 해당 설정이 로드됨.
- 내부 알고리즘 표기:
  - `algo.name: a2c_continuous`
  - `config.ppo: True`
- 즉, 구현상 `rl_games`의 A2C continuous 러너를 사용하지만
  실제 업데이트 규칙은 PPO 클리핑(`e_clip`, `mini_epochs`, `normalize_advantage`) 기반.

### 9-2. 주요 에이전트/모델 타입 (현재 코드 기준)
- `a2c_continuous`:
  - 연속 action 공간용 on-policy 에이전트.
  - `horizon_length` 단위 rollout 후 PPO-style 갱신 수행.
- `continuous_a2c_logstd`:
  - 정책이 Gaussian action을 출력하는 actor-critic 모델.
  - `fixed_sigma=True`일 때 초기 `sigma`를 고정값 기반으로 사용.
- `actor_critic` (single-head):
  - 현재 `5g_lift_left-v1` 기본 네트워크.
  - actor/critic shared trunk(`separate=False`) + MLP(`[256, 128, 64]`, `elu`).
- `dualhead_a2c` (bimanual 전용):
  - 저장소에 커스텀 등록된 `rl_games` 네트워크(`sbm.rl.rl_games_networks`).
  - 좌/우 팔 관측/행동을 분리 인코딩할 때 사용.
  - 예: `pipeline/gripper/both/2g_pouring_v1`의 `rl_games_ppo_dualhead_cfg.yaml`.

### 9-3. 운영 시 유의점 (orchestrator/runner)
- `agent` 문자열이 `rl_games_`로 시작하면 `atuo/orchestrator.py`가
  학습 스크립트를 `scripts/reinforcement_learning/rl_games/train.py`로 자동 전환.
- 로그 루트는 `log/rl_games` 계열로 전환되며, 체크포인트는 보통 `nn/*.pth`.
- resume 시 `atuo/runner.py`가
  - run 디렉터리(`test*`) 탐색
  - `nn/<checkpoint>` 우선
  순으로 체크포인트를 자동 해석.
