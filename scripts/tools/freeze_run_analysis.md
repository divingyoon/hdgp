# freeze_run_analysis.py

`TensorBoard events + checkpoint 파일`을 고정 JSON으로 요약하고, 고정 프롬프트를 만들어 LLM 답변을 파일로 저장하는 도구입니다.

## 왜 쓰는가

- 매번 이벤트 파일 전체를 다시 읽는 토큰 낭비를 줄입니다.
- `snapshot.json` 기반으로 입력을 고정해 답변 변동성을 줄입니다.
- 로그가 바뀌지 않으면 기존 `gemini_answer.md`를 재사용합니다.

## 기본 사용

```bash
python3 hdgp/scripts/tools/freeze_run_analysis.py \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2 \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v1/test1 \
  --base-run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v1/test1 \
  --out-dir /home/user/rl_ws/hdgp/log/analysis_llm
```

생성 파일:

- `<out-dir>/<run_id>/snapshot.json`
- `<out-dir>/<run_id>/prompt.txt`
- `<out-dir>/<run_id>/manifest.json`
- `<out-dir>/<run_id>/compare_to_base.json` (비교 모드일 때)

## Gemini까지 자동 실행

```bash
python3 hdgp/scripts/tools/freeze_run_analysis.py \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2 \
  --base-run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v1/test1 \
  --out-dir /home/user/rl_ws/hdgp/log/analysis_llm \
  --gemini --report-verbosity long
```

인증 이슈가 있으면:

```bash
python3 hdgp/scripts/tools/freeze_run_analysis.py \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2 \
  --out-dir /home/user/rl_ws/hdgp/log/analysis_llm \
  --gemini --gemini-no-browser
```

## 캐시 재생성 강제

```bash
python3 hdgp/scripts/tools/freeze_run_analysis.py \
  --run /home/user/rl_ws/hdgp/log/rl_games/pipeline/left/5g_lift_left_v2/test2 \
  --out-dir /home/user/rl_ws/hdgp/log/analysis_llm \
  --force-refresh
```

## 리포트 길이 제어

- `--report-verbosity short`: 짧은 요약
- `--report-verbosity medium`: 중간 길이
- `--report-verbosity long`: 긴 보고서(지표/근거/실험 항목 확장)
- 미지정 기본값: `long`
