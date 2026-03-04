# DG5F Right (Tesollo) vs Allegro Hand Joint Comparison

## Scope and source files

- Tesollo source: `/home/user/rl_ws/urdf/delto_m_ros2/dg_isaacsim/dg5f_right/dg5f_right.usd`
- Requested Allegro source: `/home/user/Desktop/allegro.usd`
- Important caveat: `/home/user/Desktop/allegro.usd` does not contain the hand joints directly. It prepends a remote payload:
  - `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/IsaacLab/Robots/KukaAllegro/configuration/allegro.usd`
- Because that payload is not locally resolvable in this environment, the actual Allegro joint parameters below were extracted from the locally available surrogate asset:
  - `/home/user/rl_ws/DEXTRAH/dextrah_lab/assets/kuka_allegro/kuka_allegro.usd`
- Scope filtered for a useful hand-only comparison:
  - Tesollo: 20 revolute finger joints (`rj_dg_*_*`), excluding `root_joint` fixed joint
  - Allegro: 16 revolute hand joints (`thumb/index/middle/ring_joint_*`), excluding KUKA arm joints, mount fixed joints, and Biotac tip fixed joints

## High-level differences

- Tesollo has 5 fingers x 4 DOF = 20 revolute finger joints.
- Allegro has 4 fingers x 4 DOF = 16 revolute finger joints.
- Tesollo uses mixed joint axes (`X`, `Y`, `Z`) depending on finger and stage.
- Allegro hand joints are all authored with `physics:axis = Z`.
- Tesollo drive stiffness is low and graded by joint stage (roughly `0.0957` to `2.4536`).
- Allegro drive stiffness is extremely high and uniform (`10000000.0`) on all hand joints.
- Tesollo drive damping is very small (roughly `3.83e-05` to `9.81e-04`).
- Allegro damping is either `0.0` or `0.025` depending on finger/joint.
- Tesollo drive max force is `30.0` on all finger joints.
- Allegro drive max force is `0.35` on all hand joints.
- Tesollo max joint velocity is `418.99997` on all finger joints.
- Allegro max joint velocity is `359.98938` on all hand joints.
- `physxJoint:jointFriction` is not authored on Tesollo finger joints.
- `physxJoint:jointFriction = 0.035` is authored on all Allegro hand joints.
- Tesollo has a little finger (finger 5); Allegro has no little finger counterpart.

## Authored constants shared across movable joints

### Tesollo (`rj_dg_*_*`)

- `drive:angular:physics:type = force`
- `drive:angular:physics:targetPosition = 0.0`
- `physics:breakForce = 3.4028234663852886e+38`
- `physics:breakTorque = 3.4028234663852886e+38`
- `physics:localPos1 = [0.0, 0.0, 0.0]`
- `physics:localRot1 = {real: 1.0, imag: [0.0, 0.0, 0.0]}`
- Additional authored scalar that varies by joint: `physics:JointEquivalentInertia`

### Allegro hand (`*_joint_*`)

- `drive:angular:physics:type = force`
- `drive:angular:physics:targetPosition = 0.0`
- `physics:breakForce = 3.4028234663852886e+38`
- `physics:breakTorque = 3.4028234663852886e+38`
- `physics:localPos1 = [0.0, 0.0, 0.0]`
- `physics:localRot1 = {real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `physxJoint:jointFriction = 0.03500000014901161` on every hand revolute joint

## Finger-by-finger comparison

### Thumb

| DOF | Tesollo joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction | Allegro joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `rj_dg_1_1` | `X` | `-21.999998092651367 / 77.0` | `0.8294013142585754` | `0.0003317605296615511` | `30.0` | `418.9999694824219` | `N/A` | `thumb_joint_0` | `Z` | `15.999527931213379 / 89.9973373413086` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 2 | `rj_dg_1_2` | `Z` | `-154.99998474121094 / 0.0` | `0.6285932660102844` | `0.0002514373045414686` | `30.0` | `418.9999694824219` | `N/A` | `thumb_joint_1` | `Z` | `-18.999439239501953 / 65.99805450439453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 3 | `rj_dg_1_3` | `X` | `-90.0 / 90.0` | `0.4585948884487152` | `0.00018343795090913773` | `30.0` | `418.9999694824219` | `N/A` | `thumb_joint_2` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 4 | `rj_dg_1_4` | `X` | `-90.0 / 90.0` | `0.16977134346961975` | `6.790853512939066e-05` | `30.0` | `418.9999694824219` | `N/A` | `thumb_joint_3` | `Z` | `-15.999527931213379 / 100.99701690673828` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |

### Index

| DOF | Tesollo joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction | Allegro joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `rj_dg_2_1` | `X` | `-19.999998092651367 / 30.999996185302734` | `2.4380815029144287` | `0.00097523262957111` | `30.0` | `418.9999694824219` | `N/A` | `index_joint_0` | `Z` | `-31.999055862426758 / 31.999055862426758` | `10000000.0` | `0.02500000037252903` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 2 | `rj_dg_2_2` | `Y` | `-0.0 / 115.0` | `0.827620804309845` | `0.0003310483298264444` | `30.0` | `418.9999694824219` | `N/A` | `index_joint_1` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.02500000037252903` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 3 | `rj_dg_2_3` | `Y` | `-90.0 / 90.0` | `0.411169171333313` | `0.00016446768131572753` | `30.0` | `418.9999694824219` | `N/A` | `index_joint_2` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.02500000037252903` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 4 | `rj_dg_2_4` | `Y` | `-90.0 / 90.0` | `0.09574367105960846` | `3.829746856354177e-05` | `30.0` | `418.9999694824219` | `N/A` | `index_joint_3` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.02500000037252903` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |

### Middle

| DOF | Tesollo joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction | Allegro joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `rj_dg_3_1` | `X` | `-30.0 / 30.0` | `2.453554391860962` | `0.000981421791948378` | `30.0` | `418.9999694824219` | `N/A` | `middle_joint_0` | `Z` | `-31.999055862426758 / 31.999055862426758` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 2 | `rj_dg_3_2` | `Y` | `0.0 / 115.0` | `0.8264341354370117` | `0.00033057364635169506` | `30.0` | `418.9999694824219` | `N/A` | `middle_joint_1` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 3 | `rj_dg_3_3` | `Y` | `-90.0 / 90.0` | `0.4111796021461487` | `0.00016447184316348284` | `30.0` | `418.9999694824219` | `N/A` | `middle_joint_2` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 4 | `rj_dg_3_4` | `Y` | `-90.0 / 90.0` | `0.09574819356203079` | `3.8299276639008895e-05` | `30.0` | `418.9999694824219` | `N/A` | `middle_joint_3` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |

### Ring

| DOF | Tesollo joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction | Allegro joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `rj_dg_4_1` | `X` | `-31.999998092651367 / 15.0` | `2.309304714202881` | `0.0009237218764610589` | `30.0` | `418.9999694824219` | `N/A` | `ring_joint_0` | `Z` | `-31.999055862426758 / 31.999055862426758` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 2 | `rj_dg_4_2` | `Y` | `0.0 / 109.99999237060547` | `0.8272904753684998` | `0.0003309161984361708` | `30.0` | `418.9999694824219` | `N/A` | `ring_joint_1` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 3 | `rj_dg_4_3` | `Y` | `-90.0 / 90.0` | `0.4109422266483307` | `0.00016437689191661775` | `30.0` | `418.9999694824219` | `N/A` | `ring_joint_2` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |
| 4 | `rj_dg_4_4` | `Y` | `-90.0 / 90.0` | `0.09573269635438919` | `3.829307752312161e-05` | `30.0` | `418.9999694824219` | `N/A` | `ring_joint_3` | `Z` | `-15.999527931213379 / 98.99707794189453` | `10000000.0` | `0.0` | `0.3499999940395355` | `359.9893798828125` | `0.03500000014901161` |

### Little

| DOF | Tesollo joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction | Allegro joint | Axis | Limits (lower / upper) | Stiffness | Damping | MaxForce | MaxVel | Friction |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `rj_dg_5_1` | `Z` | `-0.0 / 60.0` | `1.7342278957366943` | `0.0006936911377124488` | `30.0` | `418.9999694824219` | `N/A` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| 2 | `rj_dg_5_2` | `X` | `-15.0 / 90.0` | `1.107469916343689` | `0.0004429879772942513` | `30.0` | `418.9999694824219` | `N/A` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| 3 | `rj_dg_5_3` | `Y` | `-90.0 / 90.0` | `0.4551452100276947` | `0.00018205808009952307` | `30.0` | `418.9999694824219` | `N/A` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| 4 | `rj_dg_5_4` | `Y` | `-90.0 / 90.0` | `0.16935937106609344` | `6.774374924134463e-05` | `30.0` | `418.9999694824219` | `N/A` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |

## Detailed joint frame inventory

This section captures the per-joint authored frame attachment data that is not practical to fit in the comparison tables: `physics:body0`, `physics:body1`, `physics:localPos0`, `physics:localPos1`, `physics:localRot0`, and `physics:localRot1`.

### Tesollo frame attachments

- `rj_dg_1_1`: `/dg5f_right/rl_dg_mount` -> `/dg5f_right/rl_dg_1_1`, localPos0=`[-0.016200000420212746, 0.01899999938905239, 0.08659999817609787]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_1_2`: `/dg5f_right/rl_dg_1_1` -> `/dg5f_right/rl_dg_1_2`, localPos0=`[0.04194999858736992, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_1_3`: `/dg5f_right/rl_dg_1_2` -> `/dg5f_right/rl_dg_1_3`, localPos0=`[0.0, 0.03099999949336052, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_1_4`: `/dg5f_right/rl_dg_1_3` -> `/dg5f_right/rl_dg_1_4`, localPos0=`[0.0, 0.03880000114440918, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_2_1`: `/dg5f_right/rl_dg_mount` -> `/dg5f_right/rl_dg_2_1`, localPos0=`[-0.0071000000461936, 0.027000000700354576, 0.13989998400211334]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_2_2`: `/dg5f_right/rl_dg_2_1` -> `/dg5f_right/rl_dg_2_2`, localPos0=`[0.017650000751018524, 0.0, 0.026499999687075615]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_2_3`: `/dg5f_right/rl_dg_2_2` -> `/dg5f_right/rl_dg_2_3`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_2_4`: `/dg5f_right/rl_dg_2_3` -> `/dg5f_right/rl_dg_2_4`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_3_1`: `/dg5f_right/rl_dg_mount` -> `/dg5f_right/rl_dg_3_1`, localPos0=`[-0.0071000000461936, 0.0024999999441206455, 0.14389999210834503]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_3_2`: `/dg5f_right/rl_dg_3_1` -> `/dg5f_right/rl_dg_3_2`, localPos0=`[0.017650000751018524, 0.0, 0.026499999687075615]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_3_3`: `/dg5f_right/rl_dg_3_2` -> `/dg5f_right/rl_dg_3_3`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_3_4`: `/dg5f_right/rl_dg_3_3` -> `/dg5f_right/rl_dg_3_4`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_4_1`: `/dg5f_right/rl_dg_mount` -> `/dg5f_right/rl_dg_4_1`, localPos0=`[-0.0071000000461936, -0.02199999988079071, 0.13589999079704285]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_4_2`: `/dg5f_right/rl_dg_4_1` -> `/dg5f_right/rl_dg_4_2`, localPos0=`[0.017650000751018524, 0.0, 0.026499999687075615]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_4_3`: `/dg5f_right/rl_dg_4_2` -> `/dg5f_right/rl_dg_4_3`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_4_4`: `/dg5f_right/rl_dg_4_3` -> `/dg5f_right/rl_dg_4_4`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_5_1`: `/dg5f_right/rl_dg_mount` -> `/dg5f_right/rl_dg_5_1`, localPos0=`[0.010300000198185444, -0.019500000402331352, 0.09200000017881393]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_5_2`: `/dg5f_right/rl_dg_5_1` -> `/dg5f_right/rl_dg_5_2`, localPos0=`[0.0, -0.02800000086426735, 0.038100000470876694]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_5_3`: `/dg5f_right/rl_dg_5_2` -> `/dg5f_right/rl_dg_5_3`, localPos0=`[0.0, 0.0, 0.03099999949336052]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `rj_dg_5_4`: `/dg5f_right/rl_dg_5_3` -> `/dg5f_right/rl_dg_5_4`, localPos0=`[0.0, 0.0, 0.03880000114440918]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`

### Allegro frame attachments

- `index_joint_0`: `/kuka_allegro/palm_link` -> `/kuka_allegro/index_link_0`, localPos0=`[0.051430199295282364, -0.036320000886917114, -0.011300000362098217]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 0.030874667689204216, imag: [0.7064487338066101, -0.030876096338033676, 0.7064160108566284]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `middle_joint_0`: `/kuka_allegro/palm_link` -> `/kuka_allegro/middle_link_0`, localPos0=`[0.05373749881982803, 0.008777099661529064, -0.011300000362098217]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 3.2760097383288667e-05, imag: [0.7071231603622437, -3.275858034612611e-05, 0.7070903778076172]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `ring_joint_0`: `/kuka_allegro/palm_link` -> `/kuka_allegro/ring_link_0`, localPos0=`[0.051430199295282364, 0.053874898701906204, -0.011300000362098217]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: -0.03080921061336994, imag: [0.7064515948295593, 0.0308106429874897, 0.7064188718795776]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `thumb_joint_0`: `/kuka_allegro/palm_link` -> `/kuka_allegro/thumb_link_0`, localPos0=`[-0.03674820065498352, -0.008128100074827671, -0.029500000178813934]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 0.7064344882965088, imag: [-0.030859023332595825, -0.7064330577850342, -0.030826270580291748]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `index_joint_1`: `/kuka_allegro/index_link_0` -> `/kuka_allegro/index_link_1`, localPos0=`[0.0, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: -0.49995365738868713, imag: [0.5000463724136353, 0.4999768137931824, 0.5000231266021729]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `index_joint_2`: `/kuka_allegro/index_link_1` -> `/kuka_allegro/index_link_2`, localPos0=`[0.05400000140070915, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `index_joint_3`: `/kuka_allegro/index_link_2` -> `/kuka_allegro/index_link_3`, localPos0=`[0.03840000182390213, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `middle_joint_1`: `/kuka_allegro/middle_link_0` -> `/kuka_allegro/middle_link_1`, localPos0=`[0.0, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: -0.49995365738868713, imag: [0.5000463724136353, 0.4999768137931824, 0.5000231266021729]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `middle_joint_2`: `/kuka_allegro/middle_link_1` -> `/kuka_allegro/middle_link_2`, localPos0=`[0.05400000140070915, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `middle_joint_3`: `/kuka_allegro/middle_link_2` -> `/kuka_allegro/middle_link_3`, localPos0=`[0.03840000182390213, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `ring_joint_1`: `/kuka_allegro/ring_link_0` -> `/kuka_allegro/ring_link_1`, localPos0=`[0.0, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: -0.49995365738868713, imag: [0.5000463724136353, 0.4999768137931824, 0.5000231266021729]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `ring_joint_2`: `/kuka_allegro/ring_link_1` -> `/kuka_allegro/ring_link_2`, localPos0=`[0.05400000140070915, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `ring_joint_3`: `/kuka_allegro/ring_link_2` -> `/kuka_allegro/ring_link_3`, localPos0=`[0.03840000182390213, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `thumb_joint_1`: `/kuka_allegro/thumb_link_0` -> `/kuka_allegro/thumb_link_1`, localPos0=`[0.004999999888241291, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 0.7071231603622437, imag: [0.7070903778076172, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `thumb_joint_2`: `/kuka_allegro/thumb_link_1` -> `/kuka_allegro/thumb_link_2`, localPos0=`[0.0, 0.0, 0.055399999022483826]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 3.2760097383288667e-05, imag: [0.7071231603622437, -3.275858034612611e-05, 0.7070903778076172]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`
- `thumb_joint_3`: `/kuka_allegro/thumb_link_2` -> `/kuka_allegro/thumb_link_3`, localPos0=`[0.05139999836683273, 0.0, 0.0]`, localPos1=`[0.0, 0.0, 0.0]`, localRot0=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`, localRot1=`{real: 1.0, imag: [0.0, 0.0, 0.0]}`

## Interpretation notes

- Tesollo is tuned as a much softer, lower-gain hand in the authored drive parameters.
- Allegro is authored like a very stiff position-holding hand, with tiny available drive force but massive stiffness.
- Tesollo thumb kinematics are more asymmetric than Allegro thumb, especially at DOF 2 (`-155 deg` to `0 deg`).
- Allegro MCP-like proximal joints (`*_joint_0`) are the only ones with non-zero damping on index; middle/ring/thumb start with zero damping.
- Tesollo little finger has no Allegro counterpart, so any controller retargeting from Tesollo to Allegro will need explicit finger-drop or remapping logic.
