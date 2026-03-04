# FABRICS Framework: High-Performance Robot Control Guide

FABRICS는 로봇 제어를 위한 비선형, 자율 2차 미분 방정식 기반의 프레임워크입니다. NVIDIA Warp와 PyTorch를 활용하여 GPU 가속을 지원하며, 복잡한 로봇 시스템의 안전하고 효율적인 제어를 가능하게 합니다.

## 1. 핵심 특징 및 장점

### 1.1 GPU 가속 및 병렬화 (High Scalability)
*   **NVIDIA Warp & CUDA Graph:** 로봇의 기구학(Kinematics) 및 제어 법칙 계산을 GPU 커널로 구현하여 수천 개의 환경을 동시에 시뮬레이션할 수 있습니다. CUDA Graph 기능을 통해 지연 시간을 최소화하고 처리량을 극대화합니다.
*   **가속화된 기구학 엔진:** `src/fabrics_sim/prod/kinematics.py`에서 볼 수 있듯이, 링크별 병렬 처리가 가능한 커널을 통해 자코비안(Jacobian) 및 공간 속도(Spatial Velocity)를 초고속으로 계산합니다.

### 1.2 수학적 안정성 및 경로 일관성 (Provable Stability)
*   **Geometric Fabrics:** 리아푸노프 안정성(Lyapunov stability)이 증명된 기하학적 제어 기법을 사용합니다. 이는 로봇이 목표 상태로 수렴하는 과정에서 발산하지 않음을 보장합니다.
*   **2차 미분 방정식 기반:** 가속도 수준의 제어를 수행하므로 하드웨어의 동역학적 특성을 잘 반영하며 부드러운 움직임을 제공합니다.

### 1.3 모듈형 설계 (Task-map Container)
*   **Task-maps:** 관절 공간(C-space)을 작업 공간(Task-space, 예: 손바닥 위치, 관절 한계 등)으로 매핑합니다.
*   **Pullback 연산:** 여러 작업 공간에서 정의된 힘(Force)과 질량(Metric)을 관절 공간으로 수학적으로 통합(Pullback & Combine)하여 최종 토크/가속도 명령을 생성합니다.

---

## 2. Isaac Lab 및 강화학습(RL) 활용 장점

### 2.1 안전한 탐색 레이어 (Safe RL Layer)
강화학습 에이전트가 직접 관절 토크를 제어하는 대신, FABRICS를 **안전 계층(Safe Layer)**으로 활용할 수 있습니다.
*   **제약 조건 강제:** 관절 한계(Joint Limits), 충돌 회피(Collision Avoidance), 특이점 회피(Singularity Avoidance)를 FABRICS가 하드 코딩된 제약 조건으로 처리합니다.
*   **에이전트 역할 단순화:** RL 에이전트는 고수준의 목표(예: 손바닥의 목표 포즈)만 학습하면 되므로 액션 공간(Action Space)이 줄어들고 학습 속도가 비약적으로 향상됩니다.

### 2.2 대규모 병렬 학습 (Vectorized Environments)
*   Isaac Lab의 벡터화된 환경(`batch_size` 지원)과 완벽하게 호환됩니다. 수천 개의 로봇 에이전트가 각각 독립적인 FABRICS 컨트롤러를 GPU에서 동시에 실행하므로 학습 효율이 극대화됩니다.

### 2.3 보상 설계의 단순화 (Reward Shaping)
*   학습 과정에서 충돌 방지나 안전에 대한 복잡한 패널티(Penalty)를 설계할 필요가 줄어듭니다. FABRICS가 기본적으로 안전을 보장하므로, 에이전트는 작업 완수(Task Success)에 더 집중할 수 있습니다.

---

## 3. Sim2Real (시뮬레이션에서 실물로의 전이) 이점

### 3.1 모델 불확실성에 대한 강건성 (Robustness)
*   FABRICS는 정밀한 마찰력이나 접촉 모델에 의존하기보다 기하학적 경로 일관성에 집중합니다. 따라서 시뮬레이션과 실제 환경 사이의 동역학적 차이(Domain Gap)가 있어도 상대적으로 안정적인 전이가 가능합니다.

### 3.2 반응형 제어 (Reactive Control)
*   실시간으로 환경 변화(예: 움직이는 장애물)에 반응하는 특성을 가집니다. 시뮬레이션에서 학습되지 않은 돌발 상황에서도 FABRICS의 척력(Repulsion) 항이 작동하여 실제 로봇의 파손을 방지합니다.

### 3.3 가속도 및 저크 제한 (Accel/Jerk Limiting)
*   `src/fabrics_sim/fabrics/fabric.py`에는 가속도와 저크(Jerk)를 제한하는 커널이 포함되어 있습니다. 이는 실제 로봇 하드웨어의 모터 및 감속기에 무리가 가지 않는 물리적으로 실행 가능한 명령을 생성하도록 돕습니다.

---

## 4. 실제 사용 예시 (Kuka-Allegro Robot)

제공된 예제(`examples/kuka_allegro_pose_fabric_example.py`)를 통해 다음과 같은 워크플로우를 구현할 수 있습니다:

1.  **환경 설정:** `WorldMeshesModel`을 통해 충돌체를 로드합니다.
2.  **컨트롤러 생성:** `KukaAllegroPoseFabric`을 생성하여 로봇의 기구학 및 제어 로직을 로드합니다.
3.  **목표 설정:** 에이전트나 사용자가 손바닥의 목표 포즈(`palm_target`)와 손가락의 PCA 목표(`hand_targets`)를 전달합니다.
4.  **실행:** `integrator.step()`을 통해 매 타임스텝마다 안전이 보장된 다음 상태(q, qd)를 계산합니다.

---

## 5. 결론

FABRICS는 **"안전성(Safety)"**과 **"성능(Performance)"**을 동시에 잡아야 하는 현대 로봇 제어 및 강화학습 연구에 최적화된 도구입니다. 특히 Isaac Lab과 결합했을 때 대규모 병렬 학습의 이점을 극대화하면서도, 실물 로봇으로의 전이 시 안전성을 보장할 수 있는 강력한 솔루션을 제공합니다.
