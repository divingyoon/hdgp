# open-tesol_r_reach_v5 개발 및 디버깅 종합 리포트

---

## 1. 개발 배경 및 목적 (Background & Goal)

기존 `reach_v4` 및 `grasp_v1` 환경은 도달(Reach)뿐만 아니라 파지(Grasp), 리프팅(Lift), 접촉 센서(Contact Sensor), 손가락 관절 제어, 외란(Disturbance) 등 복합적인 기능이 결합되어 있어, **순수한 접근 및 정렬(Pure Side Reach & Alignment)** 단계의 문제점을 독립적으로 격리하고 디버깅하기 어려웠습니다.

`reach_v5`는 다음과 같은 목표로 완전히 새로 구축(Clean-Slate)되었습니다:
1. **Clean-Slate 아키텍처**: 불필요한 파지/리프트/센서 의존성을 100% 제거하고, 로봇 손바닥(Palm)의 6D 궤적 추종(Fabrics IK) 및 공간 정렬에 집중.
2. **6D End-Effector Control**: 팔 7-DoF는 Fabrics IK를 통해 손바닥 6D 델타 포즈로 제어하며, Delto 5지 핸드(20-DoF)는 사전 파지 접근 자세(`HAND_APPROACH_POSE`)로 고정.
3. **MultiAsset 8종 컵 대응**: 테이블 위 다양한 규격의 컵에 대해 균일하고 안정적인 측면 접근(Side Approach) 정책 학습.

---

## 2. 환경 규격 및 아키텍처 (Specifications)

### 2.1 제어 및 관측 공간

| 항목 | 차원 | 구성 세부 내용 |
|---|---|---|
| **Action Space** | **6D** | 손바닥 6D 델타 목표 ($\Delta X, \Delta Y, \Delta Z, \Delta \text{Roll}, \Delta \text{Pitch}, \Delta \text{Yaw}$) $\to$ Fabrics IK $\to$ 팔 7-DoF 제어 |
| **Observation Space** | **37D** | • 팔 관절 위치 (`arm_joint_pos`, 7D)<br>• 팔 관절 속도 (`arm_joint_vel * 0.1`, 7D)<br>• 손바닥 3D 위치 (Local, 3D)<br>• 손바닥 쿼터니언 (Local, 4D)<br>• 컵 3D 위치 (Local, 3D)<br>• 컵 쿼터니언 (Local, 4D)<br>• 손바닥 $\to$ 컵 상대 벡터 (`cup_pos - palm_pos`, 3D)<br>• 직전 액션 (`last_actions`, 6D) |
| **Critic Obs Space** | **37D** | `asymmetric_obs = False` (Observation과 동일) |
| **Episode Length** | **200 steps** | $3.33\text{s}$ @ $60\text{Hz}$ ($120\text{Hz}$ PhysX, `decimation=2`) |

---

## 3. 주요 문제 분석 및 해결 과정 (Issues & Root Causes)

`reach_v5` 초기 구축 후 실행 및 학습 과정에서 발견된 **3대 주요 문제**와 해결 과정입니다.

---

### 🚨 이슈 1: Playback 시 다중 로봇 스폰 및 컵 튕김/이탈 현상

#### 증상:
- `./play.sh` 실행 시 단일 로봇이 아닌 4대의 로봇이 나란히 스폰됨.
- 컵이 공중에 뜨거나 스폰 즉시 바닥으로 튕겨나가며 에피소드가 1스텝 만에 강제 리셋됨.

#### 원인 분석:
1. **IsaacLab CLI 기본값**: IsaacLab의 `play.py`는 기본 `--num_envs` 인자가 없을 때 4개 환경을 기본 생성함.
2. **원점 좌표계 오프셋 누락**: 컵 리셋 로직(`reset_idx`)에서 `self.scene.env_origins[env_ids]`가 누락되어 모든 환경(env 0~3)의 컵이 월드 원점의 동일한 위치 `(0.18, 0.40, 0.0)`에 겹쳐서 생성됨 $\to$ 강한 물리 충돌(Contact Explosion) 발생.
3. **관측치 좌표계 불일치**: 손바닥 위치(`palm_pos_w`)와 컵 위치(`cup_pos_w`)가 환경별 로컬 좌표가 아닌 글로벌 월드 좌표로 정책 네트워크에 전달되어 다중 환경에서 관측치 왜곡 발생.

#### 해결 조치:
1. **로컬 좌표계 일원화**:
   ```python
   # grasp_right_env.py
   self.palm_pos = self.palm_pos_w - self.scene.env_origins
   self.object_pos = self.object_pos_w - self.scene.env_origins
   ```
2. **컵 리셋 시 원점 오프셋 적용**:
   ```python
   world_cup_pos = self.cup_init_pos[env_ids] + self.scene.env_origins[env_ids]
   ```
3. **`play.sh` 래퍼 기본값 1 적용**:
   `--num_envs` 옵션 미지정 시 기본 1개 환경만 시각화하도록 래퍼 스크립트 작성.

---

### 🚨 이슈 2: 로봇이 초기 자세에서 전혀 움직이지 않고 멈춰 있는 현상 (Freezing / Idling)

#### 증상:
- 학습을 진행해도 로봇이 팔을 뻗지 않고 홈 자세(Home Pose)에서 미세하게 떨거나 가만히 대기함.

#### 원인 분석 (보상 지형 분석):
1. **지나치게 좁은 가우시안 커널 ($\sigma=0.05\text{m}$)**:
   - 홈 포즈에서 컵까지의 초기 거리는 약 $18\text{cm}\sim25\text{cm}$.
   - $\exp\left(-\frac{0.18^2}{2 \times 0.05^2}\right) = \exp(-6.48) \approx 0.0015$ 로 보상 기울기(Gradient)가 거의 0에 수렴하여 에이전트가 탐색 동기를 상실함.
2. **정렬 보상의 "무임승차(Free Lunch)" 현상**:
   - 손바닥 정면 정렬 보상($r_{\text{align}}$)이 초기 홈 자세에서도 $+1.0$ 만점을 제공함.
   - 에이전트는 팔을 움직여 컵에 다가가려다 실패해 행동 페널티를 받느니, **가만히 서서 정렬 보상(+1.0)만 안전하게 챙기는 국소 최적해(Local Optimum)**에 수렴함.

#### 해결 조치:
가장 안정적으로 도달 학습에 성공했던 **`reach_v3`의 직교 분리(Orthogonal Decoupled) 보상 체계**로 전면 개편:
- **수평 접근 (XY Plane)**: $d_{\text{standoff}} = 8\text{cm}$ 목표 지점까지의 선형+Tanh 연속 보상 ($w=0.30$).
- **수직 높이 (Z Axis)**: 컵 중심 대비 $+4\text{cm}$ 높이 도달 보상 ($w=0.30$).
- **3D 공간 이중 축 정렬**: 손바닥 법선(+X)이 컵 중심을 향하도록 유도 ($w=0.20$) + 손가락(+Z)이 지면 아래를 향하도록 유도 ($w=0.10$).
- **컵 외란 페널티**: 도달 전에 컵을 밀치거나 넘어뜨리면 즉시 강한 감점 부여.

#### 보상 경사도 검증:
- 초기 홈 자세 (컵과 18cm 거리): 종합 스텝 보상 **`-0.180`** (음수 페널티로 정지 상태 억제)
- 컵 전방 8cm 스탠드오프 도달: 종합 스텝 보상 **`+0.476`** (강한 양의 경사도 형성)

---

### 🚨 이슈 3: `play.sh` 실행 시 최신 체크포인트가 아닌 과거 체크포인트가 로드되는 현상

#### 증상:
- 새로 학습을 시작했음에도 `play.sh --use_last_checkpoint` 실행 시 이전에 정지 상태로 끝난 `test1`의 구버전 가중치가 로드됨.

#### 원인 분석:
- `rl_games_ppo_cfg.yaml`의 `params.config.full_experiment_name`이 `'test1'`으로 하드코딩되어 있어, RL-Games의 체크포인트 탐색기가 다른 이름이나 타임스탬프로 생성된 최신 실험 폴더를 감지하지 못함.

#### 해결 조치:
- `full_experiment_name`을 `'.*'` 정규식 패턴으로 변경하여 타임스탬프 기반 최신 폴더를 우선 탐색하도록 수정.

---

## 4. 개편된 보상 함수 수식 (Decoupled Reward System)

$$R_t = R_{\text{approach\_xy}} + R_{\text{approach\_z}} + R_{\text{align\_x}} + R_{\text{align\_z}} + R_{\text{success}} - P_{\text{action}} - P_{\text{disturb}}$$

### 세부 수식:

1. **XY 평면 수평 접근 보상 ($R_{\text{approach\_xy}}$)**:
   $$e_{xy} = |\|p_{\text{palm}, xy} - p_{\text{cup}, xy}\| - d_{\text{standoff}}| \quad (d_{\text{standoff}} = 0.08\text{m})$$
   $$R_{\text{approach\_xy}} = 0.30 \times \left(1.0 - \tanh(5.0 \times e_{xy})\right)$$

2. **Z축 수직 높이 보상 ($R_{\text{approach\_z}}$)**:
   $$e_z = |p_{\text{palm}, z} - (p_{\text{cup}, z} + 0.04)|$$
   $$R_{\text{approach\_z}} = 0.30 \times \left(1.0 - \tanh(5.0 \times e_z)\right)$$

3. **손바닥 법선 정면 정렬 보상 ($R_{\text{align\_x}}$)**:
   $$\mathbf{v}_{\text{target}} = \frac{p_{\text{cup}} - p_{\text{palm}}}{\|p_{\text{cup}} - p_{\text{palm}}\|}, \quad \mathbf{n}_{\text{palm}} = \mathbf{R}_{\text{palm}} \cdot [1, 0, 0]^T$$
   $$R_{\text{align\_x}} = 0.20 \times \max\left(0, \; \mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{target}}\right)$$

4. **손가락 하방 정렬 보상 ($R_{\text{align\_z}}$)**:
   $$\mathbf{z}_{\text{palm}} = \mathbf{R}_{\text{palm}} \cdot [0, 0, 1]^T, \quad \mathbf{d}_{\text{down}} = [0, 0, -1]^T$$
   $$R_{\text{align\_z}} = 0.10 \times \max\left(0, \; \mathbf{z}_{\text{palm}} \cdot \mathbf{d}_{\text{down}}\right)$$

5. **도달 성공 보너스 ($R_{\text{success}}$)**:
   $$e_{xy} < 2\text{cm} \;\land\; e_z < 2\text{cm} \implies +2.0$$

6. **페널티 항**:
   - 액션 크기 페널티: $-0.005 \|a_t\|^2$
   - 액션 변화율 페널티: $-0.01 \|a_t - a_{t-1}\|^2$
   - 컵 외란 페널티: $\Delta p_{\text{cup}} > 1\text{cm} \implies -1.0 \times \|\Delta p_{\text{cup}}\|$

---

## 5. 실행 및 모니터링 가이드

### 🚀 학습 실행 (Background)
```bash
./train.sh open-tesol_r_reach_v5 reach_run1 --num_envs 1024
```

### 🎮 실시간 화면 시각화 (GUI Play)
```bash
# 최신 체크포인트를 단일 환경 GUI로 재생
./play.sh open-tesol_r_reach_v5 --use_last_checkpoint

# 특정 체크포인트 지정 재생
./play.sh open-tesol_r_reach_v5 --checkpoint log/rl_games/open-tesol/right/reach_v5/reach_run1/nn/open-tesol_r_reach_v5.pth
```

### 🎥 비디오 자동 녹화 (Headless Video Export)
```bash
./play.sh open-tesol_r_reach_v5 --use_last_checkpoint --headless --video --video_length 200
```
녹화된 영상은 `videos/` 디렉토리에 `.mp4` 형식으로 자동 저장됩니다.
