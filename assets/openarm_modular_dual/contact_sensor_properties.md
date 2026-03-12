---

# Isaac Sim 컨택 센서 설정 사양서 (RL 학습용)

## 1. 프림(Prim) 계층 구조 및 경로

센서는 각 손가락 팁의 충돌체(Mesh) 데이터를 감시하도록 설계되었습니다.

* **기본 경로 규칙**: `/openarm_dual_modular/rl_dg_{finger_id}_4/rl_dg_{finger_id}_tip/`
* **센서 프림**: `Contact_Sensor` (IsaacContactSensor)
* **대상 메쉬**: `.../collisions/rl_dg_{finger_id}_tip_c/mesh`

| 손가락 ID | 컨택 센서 절대 경로 | 감시 대상 메쉬 (contactGeoms) |
| --- | --- | --- |
| **Finger 1** | `/openarm_dual_modular/rl_dg_1_4/rl_dg_1_tip/Contact_Sensor` | `.../rl_dg_1_tip_c/mesh` |
| **Finger 2** | `/openarm_dual_modular/rl_dg_2_4/rl_dg_2_tip/Contact_Sensor` | `.../rl_dg_2_tip_c/mesh` |
| **Finger 3** | `/openarm_dual_modular/rl_dg_3_4/rl_dg_3_tip/Contact_Sensor` | `.../rl_dg_3_tip_c/mesh` |
| **Finger 4** | `/openarm_dual_modular/rl_dg_4_4/rl_dg_4_tip/Contact_Sensor` | `.../rl_dg_4_tip_c/mesh` |
| **Finger 5** | `/openarm_dual_modular/rl_dg_5_4/rl_dg_5_tip/Contact_Sensor` | `.../rl_dg_5_tip_c/mesh` |

---

## 2. 물리적 임계치 및 Saturation 설정 (Physics Attributes)

학습 시 이상치(Outlier) 제거 및 실제 하드웨어 제약 사항을 반영하기 위해 물리 엔진 단계에서 데이터를 제한합니다.

* **Threshold Min (X)**: `0.0` (모든 미세 접촉 감지)
* **Threshold Max (Y)**: `30.0` (**30N 이상은 30N으로 포화/Saturation**)

---

## 3. 통신 및 데이터 규격 (Custom Raw USD Properties)

실제 로봇 제어 환경(ROS2)과의 호환성을 위해 정의된 커스텀 속성입니다.

| 속성 이름 (Custom Attr) | 설정값 | 비고 |
| --- | --- | --- |
| `sensor_spec:sensor_code` | `0x05` | 센서 식별 코드 |
| `sensor_spec:force_limit_N` | `30.0` | 학습 시 Normalization 기준값 |
| `sensor_spec:torque_limit_mNm` | `250.0` | 6축 대응용 리미트 (시뮬레이션은 F 위주) |
| `sensor_spec:force_resolution` | `0.1N (data x 10)` | 정수형 변환 배율 |
| `sensor_spec:byte_order` | `Big Endian` | 하드웨어 통신 순서 |
| `sensor_spec:data_order` | `fx, fy, fz, tx, ty, tz` | 데이터 패킹 순서 |

---

## 4. 학습 환경(RL) 활용 가이드

### Observation 구성 (입력 데이터)

* **데이터 소스**: 각 `Contact_Sensor` 프림의 `contactForce` 속성값 (3축 $F_x, F_y, F_z$).
* **정규화(Normalization)**:

$$Observation = \frac{Raw\_Force(N)}{30.0}$$


* 결과값 범위: `[-1.0, 1.0]` (네트워크 입력에 최적화)


* **Torque 처리**: 시뮬레이션 특성상 토크 값은 무시하거나 0으로 채워 6축 형식을 유지함.

### Sim-to-Real 고려사항

* 시뮬레이션의 `threshold` 상한선(30N) 설정을 통해, 학습된 에이전트가 실제 센서의 측정 한계 범위 내에서 동작하도록 강제함.

---