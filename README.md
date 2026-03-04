# hdgp — Hand Dexterous Grasping Pipeline

OpenArm (7 DOF) + Teosllo (20 DOF) 기반 dexterous grasping RL 학습 환경.
Geometric Fabrics (DEXTRAH 방식) 를 사용하며 Isaac Lab + Isaac Sim 5.1.0 위에서 동작합니다.

---

## 전제 조건

`~/rl_ws/` 아래에 다음이 설치되어 있어야 합니다.

| 경로 | 내용 |
|------|------|
| `~/rl_ws/IsaacLab/` | Isaac Lab (Isaac Sim 5.1.0 연결) |
| `~/rl_ws/FABRICS/` | Geometric Fabrics (`fabrics_sim` 패키지) |
| `~/rl_ws/DEXTRAH/` | DextrAH 태스크 라이브러리 |
| `~/rl_ws/hdgp/` | 본 레포지토리 |

---

## 설치

### 1. 레포지토리 클론

```bash
cd ~/rl_ws
git clone <hdgp-repo-url> hdgp
```

### 2. isaaclab.sh에 경로 추가

`~/rl_ws/IsaacLab/isaaclab.sh` 파일의 초기 설정 블록(약 30~45번째 줄)에 아래 내용을 추가합니다.

```bash
# hdgp/source/openarm → openarm 패키지
HDGP_CANDIDATE="${ISAACLAB_PATH}/../hdgp"
if [[ -d "${HDGP_CANDIDATE}/source/openarm" ]]; then
    export PYTHONPATH="${HDGP_CANDIDATE}/source/openarm:${PYTHONPATH}"
fi

# FABRICS/src → fabrics_sim 패키지
FABRICS_CANDIDATE="${ISAACLAB_PATH}/../FABRICS"
if [[ -d "${FABRICS_CANDIDATE}/src" ]]; then
    export PYTHONPATH="${FABRICS_CANDIDATE}/src:${PYTHONPATH}"
fi
```

### 3. Isaac Sim Python에 의존 패키지 설치

Isaac Sim의 내장 Python(`python.sh`)에 다음 패키지를 설치합니다.

```bash
# urdfpy 및 의존 패키지 설치 (fabrics_sim 요구사항)
~/rl_ws/IsaacLab/_isaac_sim/python.sh -m pip install lxml urdfpy
```

#### 3-1. networkx 2.2 Python 3.11 호환 패치

`urdfpy`가 요구하는 `networkx 2.2`는 Python 3.11과 일부 호환되지 않으므로
아래 스크립트로 패치합니다 (FABRICS에 포함된 패치 스크립트 활용).

```bash
# Isaac Sim Python의 networkx 설치 위치를 찾아 패치
PYTHON=~/rl_ws/IsaacLab/_isaac_sim/kit/python/bin/python3
NX_DIR=$($PYTHON -m pip show networkx | grep "Location:" | awk '{print $2}')
BASE="$NX_DIR/networkx"

sed -i "s|from collections import Mapping|from collections.abc import Mapping|g" "$BASE/classes/graph.py"
sed -i "s|from collections import Mapping|from collections.abc import Mapping|g" "$BASE/classes/coreviews.py"
sed -i "s|from collections import Mapping, Set, Iterable|from collections.abc import Mapping, Set, Iterable|g" "$BASE/classes/reportviews.py"
sed -i "s|from fractions import gcd|from math import gcd|g" "$BASE/algorithms/dag.py"
sed -i "/$( echo 'from collections import defaultdict, Mapping, Set' )/c\\from collections.abc import Mapping, Set\nfrom collections import defaultdict" "$BASE/algorithms/lowest_common_ancestors.py"
sed -i "s|(np.int, \"int\"), (np.int8, \"int\"),|(int, \"int\"), (np.int8, \"int\"),|g" "$BASE/readwrite/graphml.py"

URDFPY_DIR=$($PYTHON -m pip show urdfpy | grep "Location:" | awk '{print $2}')
sed -i "s|value = np.asanyarray(value).astype(np.float)|value = np.asanyarray(value).astype(float)|g" "$URDFPY_DIR/urdfpy/urdf.py"
```

#### 3-2. warp 다운그레이드

Isaac Sim 5.1.0의 기본 `warp 1.10`은 `warp.sim` 모듈이 제거되어 FABRICS와 호환되지 않습니다.
`warp 1.8.1`로 다운그레이드합니다.

```bash
~/rl_ws/IsaacLab/_isaac_sim/python.sh -m pip install "warp-lang==1.8.1"
```

### 4. openarm 패키지 설치 (선택)

일반 Python 환경에서 `openarm` 패키지를 직접 import할 경우에만 필요합니다.
학습은 `isaaclab.sh`를 통해 실행하므로 생략 가능합니다.

```bash
cd ~/rl_ws/hdgp/source/openarm
pip install -e .
```

---

## 학습 실행

```bash
cd ~/rl_ws/IsaacLab

# 단일 GPU (헤드리스)
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
    --task "5g_grasp_right-v1" \
    --headless \
    --num_envs 2048

# 환경 수 줄여서 빠른 테스트
./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/train.py \
    --task "5g_grasp_right-v1" \
    --headless \
    --num_envs 64
```

로그는 `hdgp/log/rl_games/pipeline/right/5g_grasp_right_v1/` 에 저장됩니다.

---

## 디렉토리 구조

```
hdgp/
├── assets/                         # USD 에셋 (로봇, 물체, 씬)
│   ├── openarm_modular_dual/       # OpenArm bimanual + Teosllo right hand
│   ├── cup_bead/                   # 컵 USD
│   └── scene_objects/              # 테이블 등 씬 오브젝트
├── scripts/
│   └── reinforcement_learning/
│       └── rl_games/
│           └── train.py            # 학습 진입점
└── source/
    └── openarm/
        └── openarm/
            └── tasks/
                └── manager_based/
                    └── openarm_manipulation/
                        └── pipeline/
                            └── hand/
                                └── right/
                                    └── 5g_grasp_right_v1/   # 파지 태스크
```

---

## 주요 태스크: 5g_grasp_right_v1

**OpenArm (7 DOF) + Teosllo right hand (20 DOF) 단일 컵 파지**

| 항목 | 내용 |
|------|------|
| 액션 | 7D = 6D palm pose + 1D finger interpolation |
| 관측 | 143D (joint pos/vel 27×2, hand FK 18, object 10, goal 3, action 7, fabric q/qd 27×2) |
| 제어 | Geometric Fabrics → arm 7DOF joint target |
| 손가락 | open/grasp 포즈 선형 보간 (PCA 없음) |
| 리워드 | hand_to_object + object_to_goal + lift + finger_curl_reg |

---

## 외부 의존성 요약

| 패키지 | 경로 | 용도 |
|--------|------|------|
| FABRICS | `~/rl_ws/FABRICS/src/` | Geometric Fabrics 제어 |
| DEXTRAH | `~/rl_ws/DEXTRAH/` | 태스크 설계 참고 (직접 의존 없음) |
| Isaac Lab | `~/rl_ws/IsaacLab/` | 시뮬레이션 프레임워크 |
| Isaac Sim | `~/rl_ws/IsaacLab/_isaac_sim/` | 물리 시뮬레이터 (v5.1.0) |

> **FABRICS 내부에서 사용하는 파일**
> - `fabrics_sim/models/robots/urdf/openarm_tesollo/openarm_tesollo.urdf` — arm FK용 URDF
> - `fabrics_sim/worlds/open_tesollo_boxes.yaml` — 충돌 회피 world 정의
