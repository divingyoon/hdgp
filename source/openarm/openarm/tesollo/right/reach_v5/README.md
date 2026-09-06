# open-tesol_r_reach_v5 태스크 아키텍처 및 실행 가이드

## 1. 개요 (Task Overview)
- **로봇 자산**: `openarm_tesollo_bi_s_rl.usd` (우팔 7-DOF OpenArm + 20-DOF Delto DG-5FS 5지 핸드)
- **태스크 목표**: 테이블 위 8종 MultiAsset 컵/물체에 대한 순수 3차원 측면 접근(Side Approach) 및 정밀 법선(+X) 정렬
- **설계 철학**: 과거 `reach_v4` 및 `grasp_v1`의 파지/리프트/센서/외란 레거시 코드를 100% 배제하고 공식 강화학습 표준 가이드라인에 맞추어 밑바닥부터 모듈화 구축된 Clean-Slate 환경

---

## 2. 규격 및 공간 정의 (Spaces & Dimensions)

| 구분 | 차원 | 구성 세부 내용 |
|---|---|---|
| **Action Space** | **6D** | 손바닥 6D 목표 포즈 ($\Delta X, \Delta Y, \Delta Z, \Delta \text{Roll}, \Delta \text{Pitch}, \Delta \text{Yaw}$) $\to$ Fabrics IK $\to$ 팔 7-DoF 제어<br>*(손가락 20-DoF는 `HAND_APPROACH_POSE`로 고정)* |
| **Observation Space** | **37D** | • 팔 관절 위치 (`arm_joint_pos`, 7D)<br>• 팔 관절 속도 (`arm_joint_vel * 0.1`, 7D)<br>• 손바닥 3D 위치 (`palm_pos_w`, 3D)<br>• 손바닥 쿼터니언 (`palm_quat_w`, 4D)<br>• 컵 3D 위치 (`cup_pos_w`, 3D)<br>• 컵 쿼터니언 (`cup_quat_w`, 4D)<br>• 손바닥 $\to$ 컵 상대 벡터 (`cup_pos - palm_pos`, 3D)<br>• 직전 액션 (`last_actions`, 6D) |
| **Episode Length** | **200 steps** | $3.33\text{s}$ @ $60\text{Hz}$ ($120\text{Hz}$ PhysX, `decimation=2`) |

---

## 3. 보상 함수 (Decoupled Orthogonal Reward Formulation)

$$R_t = R_{\text{approach\_xy}} + R_{\text{approach\_z}} + R_{\text{align\_x}} + R_{\text{align\_z}} + R_{\text{success}} - P_{\text{action}} - P_{\text{disturb}}$$

1. **수평(XY) 접근 보상**:
   $$R_{\text{approach\_xy}} = 0.30 \times \left(1.0 - \tanh(5.0 \times |\|p_{\text{palm}, xy} - p_{\text{cup}, xy}\| - 0.08|)\right)$$
2. **수직(Z) 높이 보상**:
   $$R_{\text{approach\_z}} = 0.30 \times \left(1.0 - \tanh(5.0 \times |p_{\text{palm}, z} - (p_{\text{cup}, z} + 0.04)|)\right)$$
3. **손바닥 법선 정면 정렬 보상**:
   $$R_{\text{align\_x}} = 0.20 \times \max\left(0, \; \mathbf{n}_{\text{palm}} \cdot \mathbf{v}_{\text{target}}\right)$$
4. **손가락 하방 정렬 보상**:
   $$R_{\text{align\_z}} = 0.10 \times \max\left(0, \; \mathbf{z}_{\text{palm}} \cdot [0, 0, -1]^T\right)$$
5. **성공 판정**: 수평 오차 $<2\text{cm}$ 및 높이 오차 $<2\text{cm}$ 동시 만족 시 $+2.0$ 보너스 부여.

---

## 4. 실행 및 학습 가이드

### 🚀 강화학습 훈련 (GPU 환경)
```bash
# 기본 훈련 실행 (W&B 실시간 동기화)
./train.sh open-tesol_r_reach_v5 test1 --num_envs 1024

# 비디오 렌더링 포함 훈련
./train.sh open-tesol_r_reach_v5 test1 --num_envs 1024 --video
```

### 🎮 정책 롤아웃 및 비디오 시각화 (Play)
```bash
isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task open-tesol_r_reach_v5 \
    --checkpoint log/rl_games/open-tesol/right/reach_v5/test1/nn/open-tesol_r_reach_v5.pth \
    --num_envs 1 \
    --headless \
    --video \
    --video_length 200
```
