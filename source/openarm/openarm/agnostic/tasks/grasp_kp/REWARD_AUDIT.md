=== Reward Audit === (2026-09-06, grasp_kp/grasp_fj 신설 보상 — reward-audit 스킬)

변경 대상: grasp_s2r 15항 접촉 기반 보상 → progress-only 8항
  fingertip_progress 50 · lift 20·clamp(0.05+dz,0,0.5) · lift_bonus 300(1회, dz>0.10) · keypoint_progress 200(lifted 후)
  · goal_bonus 100/step(near_goal) · arm_vel −0.03·Σ|q̇| · hand_vel −0.003·Σ|q̇| · hand_floor −min(10·relu(0.215−z_min), 5)
근거: 사용자 지시(접촉센서 미사용·SimToolReal 식 전면 개편) + reports/simtoolreal_적용성_평가.md §5.2

Check 1 (Local Min): ✓(조건부) — 리프트 전 상수 수입 1.0/step(20×0.05)이 "가만히 있기" 기저선을 만든다. 그러나 접근 진행분
  (50×Σ(d0−d_min) ≈ 75) > 75 step 대기, 리프트 보너스 300 + 목표 보너스 100/step 이 압도한다. 2,048~4,096 env 에서 절벽(래치 0.10)을
  넘는 탐색이 부족할 위험은 보고서 §10 에 이미 적힌 대로 `stage/lifted`·`task/lifted_frac` 로 감시한다(500 epoch 내 상승 없으면 재검토).
Check 2 (Hacking):   ✗→수정 — goal_bonus 는 lifted 게이트가 없다(SimToolReal 동일). 첫 목표 z 하한 0.12 와 시작 허용오차 0.06 이면
  dz=0.06 에서 near_goal 이 성립해 **래치(0.10)를 안 넘고도 100/step** 을 받는 구멍이 있다. 또 이후 목표 박스 z 하한 0.08 도 tol 0.06 과
  겹치면 테이블 근처 밀기가 근접 판정을 받을 수 있다.
  → 첫 목표 z 범위 (0.12,0.20) → **(0.16, 0.24)**: near_goal(tol 0.06) ⇒ dz ≥ 0.10 = 래치. 목표 박스 z 하한 0.08 → **0.10**.
  종료를 유도하는 벌점은 없다(hand_floor 는 상한 5, 리프트 전 순수입은 양수) → 조기 종료 최적화 없음.
Check 3 (Grasp):     ✓ — 접촉 항이 없어 파지와 상충하는 gradient 가 없다. 파지 품질은 보상이 아니라 리프트 후 질량정규화 외란(≈2 g)과
  낙하 종료(수입 상실)가 강제한다. hand_vel −0.003 은 SimToolReal 값 그대로(폐쇄 속도 0.005/step 리미터가 이미 더 강한 제약).
Check 4 (기존 파괴): ✓ — 신설 폴더(grasp_kp/grasp_fj). grasp_s2r 은 무변경(기준선 보존).
Check 5 (측정):      ✓ — reward/<8항>·reward/total, task/kp_dist, task/near_goal, task/lifted_frac, task/just_lifted, task/successes_mean,
  task/tol, task/hand_floor_depth_max, stage/{lifted,goal1,goal2,goal3}, done/*. 항별 로깅이 전부 TFEvents 에 남는다.

판정: REVISE → 위 2 개 기본값 수정 후 ACCEPT (DESIGN.md §2 반영, cfg 기본값 동일 적용)

예상 지표 이동:
  → stage/lifted: 0 → 0.3+ (≤500 epoch; 접근 진행분+리프트 보너스가 유일한 초기 경사)
  → task/successes_mean: 0 → ≥1 (lifted 직후; 첫 목표가 래치 바로 위)
  → task/tol: 0.06 → 0.015 계단식(3000 프레임·mean successes ≥ 2 조건)
  → reward/goal_bonus 비중이 후반 총보상의 대부분 — 정상. 리프트 전 총보상 ≈ 1.0/step 이 장기 유지되면 Check 1 의 국소최적 발동.
