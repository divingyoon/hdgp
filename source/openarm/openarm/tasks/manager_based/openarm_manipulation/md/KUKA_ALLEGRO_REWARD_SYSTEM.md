# DEXTRAH Kuka Allegro Task 보상 함수 및 메커니즘 상세 분석

이 문서는 `dextrah_kuka_allegro_env.py` 및 관련 설정 파일을 바탕으로, 고자유도 핸드(Allegro Hand)와 암(Kuka IIWA)을 이용해 물체(컵 등)를 접근, 파지, 리프트하는 보상 체계와 전략을 상세히 설명합니다.

---

## 1. 보상 함수 총괄 구조 (Total Reward Structure)

전체 보상 함수는 로봇이 물체에 접근하여 안정적으로 파지하고, 목표 위치까지 들어 올리는 일련의 과정을 유도하기 위해 4가지 주요 성분의 합으로 구성됩니다.

$$R_{total} = R_{approach} + R_{goal} + R_{lift} + R_{reg\_curl}$$

---

## 2. 세부 보상 항목 분석

### 2.1 접근 보상 (Approach Reward)
로봇의 손바닥(Palm)과 4개의 손가락 끝(Fingertips)이 물체의 중심에 가까워지도록 유도하는 단계입니다.

$$R_{approach} = w_{approach} \cdot \exp(-\alpha_{approach} \cdot d_{hand\_to\_obj})$$

*   **$d_{hand\_to\_obj}$**: 손바닥 및 손가락 끝 포인트(총 5개)와 물체 중심 간의 거리 중 **최댓값(Max Distance)**을 사용합니다.
    *   *이유*: 특정 손가락만 물체에 닿는 것이 아니라, 손 전체가 물체를 감싸듯이 고르게 접근하도록 강제하기 위함입니다.
*   **설정값**:
    *   가중치($w_{approach}$): `1.0`
    *   민감도($\alpha_{approach}$): `10.0` (Sharpness)

### 2.2 목표 도달 보상 (Object to Goal Reward)
물체를 최종 목표 위치($\mathbf{g}$)로 이동시키기 위한 주 보상입니다.

$$R_{goal} = w_{goal} \cdot \exp(\alpha_{goal} \cdot \|\mathbf{p}_{obj} - \mathbf{g}\|)$$

*   **동적 가중치**: ADR(Automatic Domain Randomization)을 통해 학습 수준에 따라 $\alpha_{goal}$ 값이 `-15.0`에서 `-20.0`으로 변화하며, 목표 지점에 가까워질수록 보상이 기하급수적으로 커지게 설계되었습니다.
*   **설정값**:
    *   가중치($w_{goal}$): `5.0`

### 2.3 리프트 보상 (Lift Reward)
물체를 테이블 바닥에서 수직으로 들어 올려 목표 높이에 도달하게 하는 보상입니다. 컵과 같은 물체를 안정적으로 고정하는 데 핵심입니다.

$$R_{lift} = w_{lift} \cdot \exp(-\alpha_{lift} \cdot |p_{obj, z} - g_z|)$$

*   **학습 전략**: 초기에는 물체를 띄우는 것($w_{lift} = 5.0$)에 집중하다가, 학습이 고도화되면 가중치를 `0.0`으로 줄여 위치 정밀도($R_{goal}$)에 집중하게 합니다.
*   **설정값**:
    *   민감도($\alpha_{lift}$): `8.5`

### 2.4 손가락 정규화 (Finger Curl Regularization)
고자유도 핸드의 무질서한 움직임을 방지하고, 물체를 감싸기 유리한 기본 형태(Nominal Curled Config)를 유지하도록 하는 패널티 성격의 보상입니다.

$$R_{reg\_curl} = w_{curl} \cdot \| \mathbf{q}_{fingers} - \mathbf{q}_{curled} \|^2$$

*   **전략**: 손가락을 넓게 벌려 접근한 후 안쪽으로 굽히며(Encase) 물체를 파지하는 동작을 유도합니다.
*   **설정값**:
    *   가중치($w_{curl}$): `-0.01` ~ `-0.005` (ADR에 의해 가변적)

---

## 3. ADR(Automatic Domain Randomization) 연동

DEXTRAH는 성공률에 따라 환경의 난이도와 보상 가중치를 실시간으로 조절합니다.

*   **파라미터 스케줄링**: `adr_custom_cfg_dict`를 통해 `lift_weight`는 감소시키고, `object_to_goal_sharpness`는 강화하여 최종적으로는 정밀한 배치가 가능하도록 유도합니다.
*   **환경 변동성**: 물체의 마찰력, 질량, 초기 위치 노이즈 등을 점진적으로 증가시켜 어떤 상황에서도 안정적인 파지가 가능하게 합니다.

---

## 4. 컵(Cup) 작업 시나리오 적용 가이드

단일 물체(컵)를 대상으로 할 때 DEXTRAH 시스템의 장점은 다음과 같습니다.

1.  **외벽/내부 파지 자동 선택**: 보상 함수는 구체적인 파지 지점을 지정하지 않습니다. 대신 손가락 끝 4개와 손바닥이 물체 중심으로 모이도록 유도($R_{approach}$)하므로, 컵의 크기에 따라 외벽을 잡거나 손가락이 안쪽으로 들어가는 최적의 파지법을 로봇이 스스로 학습합니다.
2.  **안정성 검증 (Wrench Disturbance)**: 학습 중 물체에 무작위 외력(Wrench)을 가합니다. 이를 버티고 목표 높이를 유지해야 보상을 받을 수 있으므로, 단순히 컵을 건드리는 수준이 아니라 꽉 쥐는 "Robust Grasp"를 형성하게 됩니다.
3.  **성공 기준**: 물체가 목표 위치의 `0.1m` 이내에 도달하고 `2.0초` 동안 유지되면 성공으로 판정합니다.
