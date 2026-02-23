# 5G Lift Left v2 - 보상함수 상세 가이드 (LLM/Orchestrator용)

v2는 v1 보상 구조를 유지하되, **contact sensor 기반 μ/ν 판정**과 일부 안정화(패널티 cap, 추가 termination)를 적용한 버전입니다.

기준 코드:
- `hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/lift_left_env_cfg.py`
- `hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/mdp/rewards.py`
- `hdgp/source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/left/5g_lift_left_v2/config/joint_pos_env_cfg.py`

---

## 1) 전체 연결 흐름

### 1-1. 하드 트리거 (DexPour)
- `λ (approach)`
  - `λ = 1[ ||p_ee - p_target_static|| < 0.05 ]`
- `μ (grasp)`
  - `μ = λ * 1[ n_contact(sensor) >= 3 ]`
- `ν (lift)`
  - `ν = μ * 1[ z_cup >= z_init + 0.04 ]`

### 1-2. v2 센서 경로
- `left_contact_sensor` 설정:
  - `prim_path="{ENV_REGEX_NS}/Robot/tesollo_left_.*_sensor_link"`
  - `filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup"]`
  - `history_length=1`, `track_air_time=False`
- `require_filtered_contact_matrix=True`
- `μ/ν` 계산에 센서 기반 접촉 수가 사용됨 (`sensor_cfg` 전달되는 보상들)

---

## 2) v1 대비 주요 변경점

1. `thumb_tip_z`: `weight 8 -> 10`, `std 0.10 -> 0.03`
2. `synergy_tip_z`: `weight 8 -> 10`
3. `ee_descent`: `weight 10 -> 15`
4. `object_displacement_penalty`: `penalty_max=2.0` cap 추가
5. termination 추가: `cup_xy_out_of_bounds` (`max_xy_displacement=0.10`)

---

## 3) 보상 합산 수식

- raw term: `r_i`
- weighted term: `R_i = w_i * r_i`
- total step reward: `R_total = Σ_i R_i`
- `K(d,s)=1-tanh(d/s)`

---

## 4) 항목별 상세 (수식 + 활성 게이트 + weight 증감)

아래에서 v1과 수식이 동일한 항목은 동일하게 동작합니다. v2 차이는 `sensor_cfg` 경유 게이트(μ/ν)와 weight 값입니다.

### 4-1. Reaching Phase

#### `reaching_object` (`weight=8.0`)
- `r = K(||p_target_dyn-p_ee||, 0.15) * exp(-max(d_xy-d_free,0)/s_disp)`
- 증감:
  - 증가: 접근 속도↑ / cup disturbance 위험↑
  - 감소: 접근 성공률↓ 가능

#### `reaching_object_fine` (`weight=10.0`)
- `r = K(||p_target_static-p_ee||, 0.065)`
- 증감:
  - 증가: 근거리 정밀도↑ / grasp 전환 지연 가능
  - 감소: 근거리 오차↑

#### `end_effector_orientation` (`weight=4.0`)
- `r = K(|dot(z_ee,z_obj)|, 0.3)`

#### `thumb_reaching_pose` (`weight=0.5`, 게이트 `(1-λ)`)
- `r=(1-λ)*K(Σ(q_thumb-q_open)^2,1.0)`

#### `pinky_reaching_pose` (`weight=0.5`, 게이트 `(1-λ)`)
- `r=(1-λ)*K(Σ(q_pinky-q_open)^2,1.0)`

#### `synergy_reaching_pose` (`weight=0.5`, 게이트 `(1-λ)`)
- `r=(1-λ)*K(Σ(q_synergy-0)^2,5.0)`

### 4-2. Grasp Phase

#### `thumb_grasp` (`weight=15.0`, `sensor_cfg` 전달)
- 실제 raw 수식은 v1과 동일:
  - `r = λ * (K(|rad-r_cup|,0.05) - tanh(max(r_cup-rad,0)/0.01)) * z_gate`
- `sensor_cfg`는 로그/진단 및 μ 연계 경로 일관성에 사용

#### `pinky_grasp` (`weight=12.0`, `sensor_cfg` 전달)
- `r = λ * (K(|rad-r_cup|,0.05) - tanh(max(r_cup-rad,0)/0.01))`

#### `synergy_grip` (`weight=20.0`)
- `close=clamp((a_synergy+1)/2,0,1)`
- `g_surface=K(mean_k| ||p_fk_xy-p_cup_xy||-r_cup |,0.01), k={2,3,4}`
- `r=λ*g_surface*close`

#### `thumb_tip_z` (`weight=10.0`, `std=0.03`)  **v2 강화 항목**
- `r = λ * K(max(z_thumb-z_f2,0),0.03) * exp(-||p_ee_xy-p_cup_xy||/0.06)`
- 증감:
  - 증가: 엄지 높이 정렬 강제↑
  - 과증가: 다른 grasp objective 압박 가능

#### `synergy_tip_z` (`weight=10.0`) **v2 강화 항목**
- `r = λ * K(|z_f2_tip-(z_cup+0.09)|,0.06) * exp(-||p_ee_xy-p_cup_xy||/0.06)`

#### `ee_descent` (`weight=15.0`) **v2 강화 항목**
- `r = λ*(1-μ(sensor))*K(|z_ee-(z_cup+0.04)|,0.04)*exp(-||p_ee_xy-p_cup_xy||/0.06)`
- 증감:
  - 증가: grasp 전 하강 유도↑
  - 과증가: 과하강/충돌 위험

### 4-3. Lift / Goal Phase (센서 경유)

#### `lifting_object` (`weight=10.0`, `sensor_cfg`)
- `r = μ(sensor) * 1[z_cup > z_init + 0.04]`

#### `cup_lift_progress` (`weight=20.0`, `sensor_cfg`)
- `r = μ(sensor) * tanh(max(z_cup-z_init,0)/0.05)`

#### `object_goal_tracking` (`weight=20.0`, `sensor_cfg`)
- `r = ν(sensor) * K(||p_obj-p_goal||,0.3)`

#### `object_goal_tracking_fine_grained` (`weight=10.0`, `sensor_cfg`)
- `r = ν(sensor) * K(||p_obj-p_goal||,0.1)`

### 4-4. Penalty / Regularization

#### `object_displacement` (`weight=-5.0`) **v2 안정화 핵심**
- raw:
  - `p = (max(d-0.01,0)/0.02)^2`
  - `p = clamp(p, max=2.0)`
  - `p = p*((1-0.5)+0.5*g_grasp_progress)*(1-μ(sensor))`
  - `R = -5.0 * p`
- 증감:
  - 절댓값 증가: pushing 억제↑, 탐색 위축 위험
  - 절댓값 감소: pushing 증가 가능

#### `finger_normal_range` (`weight=-2.0`)
- `R = -2.0 * Σ_j violation_j`

#### `action_rate` (`weight=-1e-4`)
- `R = -1e-4 * ||a_t-a_{t-1}||^2`

#### `joint_vel` (`weight=-1e-4`)
- `R = -1e-4 * ||qdot_left||^2`

---

## 5) 비활성(현재 weight=0) 항목

- `finger_tip_to_cup` (`finger_wrap_cylinder_reward`)
- `finger_wrap_coverage` (`finger_wrap_coverage_reward`)

---

## 6) Hydra Override 가이드 (orchestrator 직접 사용)

### 6-1. weight 키
- `env.rewards.<term>.weight=<value>`
- 예시:
  - `env.rewards.thumb_tip_z.weight=12.0`
  - `env.rewards.object_displacement.weight=-6.0`

### 6-2. params 키
- `env.rewards.<term>.params.<param>=<value>`
- 예시:
  - `env.rewards.thumb_tip_z.params.std=0.025`
  - `env.rewards.ee_descent.params.target_z_offset=0.035`

### 6-3. v2에서 자주 조정하는 키
- `env.rewards.ee_descent.weight`
- `env.rewards.thumb_tip_z.weight`
- `env.rewards.synergy_tip_z.weight`
- `env.rewards.object_displacement.weight`
- `env.rewards.object_displacement.params.threshold`

---

## 7) Termination

- `cup_dropping`: `z_cup < -0.05`
- `cup_tipping`: `dot(z_cup_axis, z_world) < cos(90°)`
- `cup_xy_out_of_bounds` (v2 전용): `||p_cup_xy - p_cup_xy_init|| > 0.10`

---

## 8) 빠른 진단 규칙 (LLM용)

- reach는 잘되는데 lift 실패:
  - `ee_descent`, `cup_lift_progress` 소폭↑
- cup pushing 반복:
  - `object_displacement` 절댓값↑, `reaching_object` 소폭↓
- grasp 접촉 불안정:
  - `thumb_grasp`, `pinky_grasp`, `synergy_grip` 소폭↑
- lift 후 goal tracking 약함:
  - `object_goal_tracking(_fine)`↑
