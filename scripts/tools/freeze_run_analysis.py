#!/usr/bin/env python3
"""Create stable RL run snapshots/prompts and optionally save Gemini answers.

This script reduces repeated token usage by caching deterministic run summaries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_run_id(run_dir: Path) -> str:
    return run_dir.name


def _find_latest_event_file(run_dir: Path) -> Path:
    summary_dir = run_dir / "summaries"
    files = sorted(summary_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no event file in {summary_dir}")
    return files[-1]


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _compute_cache_key(run_dir: Path, event_file: Path) -> str:
    nn_dir = run_dir / "nn"
    ckpt_names = sorted(p.name for p in nn_dir.glob("*.pth")) if nn_dir.is_dir() else []
    payload = {
        "run_dir": str(run_dir.resolve()),
        "event_file": {
            "name": event_file.name,
            "size": event_file.stat().st_size,
            "mtime_ns": event_file.stat().st_mtime_ns,
            "sha256": _sha256_file(event_file),
        },
        "params": {
            "env_yaml_sha256": _sha256_file(run_dir / "params" / "env.yaml"),
            "agent_yaml_sha256": _sha256_file(run_dir / "params" / "agent.yaml"),
        },
        "ckpt_names_sha256": hashlib.sha256("\n".join(ckpt_names).encode("utf-8")).hexdigest(),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean(arr: list[float]) -> float | None:
    if not arr:
        return None
    return sum(arr) / len(arr)


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


@dataclass
class ScalarStats:
    count: int
    first_step: int
    last_step: int
    first_value: float
    last_value: float
    mean_all: float
    min_value: float
    max_value: float
    mean_last_n: float | None
    slope_last_n: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "mean_all": self.mean_all,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_last_n": self.mean_last_n,
            "slope_last_n": self.slope_last_n,
        }


def _read_scalar_stats(event_file: Path, tail_n: int) -> dict[str, ScalarStats]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise RuntimeError(f"tensorboard package is required: {exc}") from exc

    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    out: dict[str, ScalarStats] = {}

    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        steps = [int(e.step) for e in events]
        values = [float(e.value) for e in events]
        n = len(values)
        start = max(0, n - tail_n)
        t_steps = steps[start:]
        t_vals = values[start:]
        out[tag] = ScalarStats(
            count=n,
            first_step=steps[0],
            last_step=steps[-1],
            first_value=values[0],
            last_value=values[-1],
            mean_all=float(sum(values) / n),
            min_value=float(min(values)),
            max_value=float(max(values)),
            mean_last_n=_mean(t_vals),
            slope_last_n=_linear_slope([float(s) for s in t_steps], t_vals),
        )
    return out


_CKPT_RE = re.compile(r"_ep_(\d+)_rew_([-0-9.]+)\.pth$")


def _checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    nn_dir = run_dir / "nn"
    if not nn_dir.is_dir():
        return {"count": 0, "latest": None, "best_reward": None, "worst_reward": None}
    rows = []
    for p in sorted(nn_dir.glob("*.pth")):
        m = _CKPT_RE.search(p.name)
        if not m:
            continue
        rows.append(
            {
                "file": p.name,
                "episode": int(m.group(1)),
                "reward": float(m.group(2)),
            }
        )
    if not rows:
        return {"count": 0, "latest": None, "best_reward": None, "worst_reward": None}
    by_ep = sorted(rows, key=lambda r: r["episode"])
    by_rew = sorted(rows, key=lambda r: r["reward"])
    return {
        "count": len(rows),
        "latest": by_ep[-1],
        "best_reward": by_rew[-1],
        "worst_reward": by_rew[0],
    }


def _build_run_snapshot(run_dir: Path, event_file: Path, tail_n: int, cache_key: str) -> dict[str, Any]:
    scalar_stats = _read_scalar_stats(event_file, tail_n)
    scalar_dict = {k: v.to_dict() for k, v in scalar_stats.items()}
    return {
        "run_id": _stable_run_id(run_dir),
        "run_dir": str(run_dir.resolve()),
        "cache_key": cache_key,
        "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_file": {
            "path": str(event_file.resolve()),
            "name": event_file.name,
            "size_bytes": event_file.stat().st_size,
            "mtime_utc": dt.datetime.utcfromtimestamp(event_file.stat().st_mtime).isoformat(timespec="seconds") + "Z",
            "sha256": _sha256_file(event_file),
        },
        "params": {
            "env_yaml_sha256": _sha256_file(run_dir / "params" / "env.yaml"),
            "agent_yaml_sha256": _sha256_file(run_dir / "params" / "agent.yaml"),
        },
        "checkpoints": _checkpoint_summary(run_dir),
        "scalar_tail_n": tail_n,
        "num_scalar_tags": len(scalar_dict),
        "scalars": scalar_dict,
    }


def _common_tag_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_tags = a.get("scalars", {})
    b_tags = b.get("scalars", {})
    common = sorted(set(a_tags.keys()) & set(b_tags.keys()))
    delta = {}
    for tag in common:
        a_last = a_tags[tag].get("last_value")
        b_last = b_tags[tag].get("last_value")
        a_mean = a_tags[tag].get("mean_last_n")
        b_mean = b_tags[tag].get("mean_last_n")
        if isinstance(a_last, (int, float)) and isinstance(b_last, (int, float)):
            d_last = b_last - a_last
        else:
            d_last = None
        if isinstance(a_mean, (int, float)) and isinstance(b_mean, (int, float)):
            d_mean = b_mean - a_mean
        else:
            d_mean = None
        delta[tag] = {"delta_last_value": d_last, "delta_mean_last_n": d_mean}
    return {
        "base_run_id": a.get("run_id"),
        "target_run_id": b.get("run_id"),
        "num_common_tags": len(common),
        "tag_delta": delta,
    }


def _make_prompt(snapshot: dict[str, Any], compare: dict[str, Any] | None, report_verbosity: str) -> str:
    compare_part = json.dumps(compare, ensure_ascii=True, indent=2) if compare else "null"
    if report_verbosity == "long":
        format_part = (
            "1) 한줄 결론\n"
            "2) 핵심 지표 12개 (숫자 포함, 항목별 1줄 해석)\n"
            "3) 학습 안정성 진단 (붕괴/진동/수렴) - 근거 최소 5개\n"
            "4) base-run 대비 차이 분석 (있는 경우) - 개선/악화 각 최소 3개\n"
            "5) 다음 실험 5개 (각 실험의 기대효과/리스크/성공판정 기준)\n"
            "6) 최종 판정: PASS / NEEDS_WORK / FAIL\n"
            "7) 실행 우선순위 TODO 5줄\n"
        )
        rules_part = (
            "- 숫자는 반드시 JSON 값 그대로 사용\n"
            "- 과장 금지, 불확실하면 '데이터 부족' 명시\n"
            "- 각 섹션은 최소 3문장(또는 3개 bullet) 이상\n"
            "- 전체 답변 분량은 한국어 기준 최소 1500자 이상\n"
            "- 마지막에 신뢰도(0~1) 한 줄 표시\n"
        )
    elif report_verbosity == "short":
        format_part = (
            "1) 한줄 결론\n"
            "2) 핵심 지표 5개 (숫자 포함)\n"
            "3) 학습 안정성 진단 (붕괴/진동/수렴)\n"
            "4) 다음 실험 3개 (각 실험의 기대효과와 리스크)\n"
            "5) 최종 판정: PASS / NEEDS_WORK / FAIL\n"
        )
        rules_part = (
            "- 숫자는 반드시 JSON 값 그대로 사용\n"
            "- 과장 금지, 불확실하면 '데이터 부족' 명시\n"
            "- 마지막에 신뢰도(0~1) 한 줄 표시\n"
        )
    else:
        format_part = (
            "1) 한줄 결론\n"
            "2) 핵심 지표 8개 (숫자 포함)\n"
            "3) 학습 안정성 진단 (붕괴/진동/수렴) - 근거 최소 3개\n"
            "4) 다음 실험 4개 (각 실험의 기대효과와 리스크)\n"
            "5) 최종 판정: PASS / NEEDS_WORK / FAIL\n"
        )
        rules_part = (
            "- 숫자는 반드시 JSON 값 그대로 사용\n"
            "- 과장 금지, 불확실하면 '데이터 부족' 명시\n"
            "- 전체 답변 분량은 한국어 기준 최소 900자 이상\n"
            "- 마지막에 신뢰도(0~1) 한 줄 표시\n"
        )

    return (
        "You are an RL training analyst.\n"
        "Use ONLY the JSON data below.\n"
        "Do not guess missing data.\n\n"
        "Required output language: Korean.\n"
        "Required format:\n"
        f"{format_part}\n"
        "Rules:\n"
        f"{rules_part}\n"
        f"RUN_SNAPSHOT_JSON:\n{json.dumps(snapshot, ensure_ascii=True, indent=2)}\n\n"
        f"COMPARE_JSON:\n{compare_part}\n"
    )


def _run_gemini(prompt_text: str, gemini_cmd: str, no_browser: bool) -> str:
    env = os.environ.copy()
    if no_browser:
        env["NO_BROWSER"] = "true"
    proc = subprocess.run([gemini_cmd, "-p", prompt_text], text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gemini command failed (code={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze RL run analysis and optionally save Gemini answers.")
    p.add_argument("--run", action="append", required=True, help="Run directory path. Use this option multiple times.")
    p.add_argument("--base-run", default=None, help="Optional base run directory for comparison delta.")
    p.add_argument("--out-dir", default="analysis_llm", help="Output directory for snapshots/prompts/answers.")
    p.add_argument("--tail-n", type=int, default=200, help="Window size for mean/slope tail stats.")
    p.add_argument("--gemini", action="store_true", help="Call gemini -p with the generated prompt.")
    p.add_argument("--gemini-cmd", default="gemini", help="Gemini CLI command name.")
    p.add_argument("--gemini-no-browser", action="store_true", help="Set NO_BROWSER=true for Gemini authentication.")
    p.add_argument(
        "--report-verbosity",
        choices=["short", "medium", "long"],
        default="long",
        help="Prompt detail level for LLM answer length/structure.",
    )
    p.add_argument("--force-refresh", action="store_true", help="Ignore cache and regenerate snapshots.")
    return p.parse_args()


def _snapshot_for_run(
    run_dir: Path, out_root: Path, tail_n: int, force_refresh: bool
) -> tuple[dict[str, Any], Path, bool]:
    run_id = _stable_run_id(run_dir)
    run_out = out_root / run_id
    snapshot_path = run_out / "snapshot.json"
    event_file = _find_latest_event_file(run_dir)
    cache_key = _compute_cache_key(run_dir, event_file)

    if snapshot_path.is_file() and not force_refresh:
        old = _load_json(snapshot_path)
        if old.get("cache_key") == cache_key:
            return old, snapshot_path, True

    snapshot = _build_run_snapshot(run_dir, event_file, tail_n, cache_key)
    _write_json(snapshot_path, snapshot)
    return snapshot, snapshot_path, False


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    run_dirs = [Path(p).resolve() for p in args.run]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run dir not found: {run_dir}")

    base_snapshot = None
    base_run_id = None
    if args.base_run:
        base_dir = Path(args.base_run).resolve()
        base_snapshot, _, _ = _snapshot_for_run(base_dir, out_root, args.tail_n, args.force_refresh)
        base_run_id = base_snapshot.get("run_id")

    for run_dir in run_dirs:
        snapshot, snapshot_path, used_cache = _snapshot_for_run(run_dir, out_root, args.tail_n, args.force_refresh)
        run_id = snapshot["run_id"]
        run_out = out_root / run_id

        compare = None
        if base_snapshot is not None and run_id != base_run_id:
            compare = _common_tag_delta(base_snapshot, snapshot)
            _write_json(run_out / "compare_to_base.json", compare)

        prompt = _make_prompt(snapshot, compare, args.report_verbosity)
        prompt_path = run_out / "prompt.txt"
        _write_text(prompt_path, prompt)

        prompt_sha256 = _sha256_text(prompt)
        manifest = {
            "run_id": run_id,
            "run_dir": snapshot["run_dir"],
            "snapshot_path": _safe_rel(snapshot_path, out_root),
            "prompt_path": _safe_rel(prompt_path, out_root),
            "prompt_sha256": prompt_sha256,
            "report_verbosity": args.report_verbosity,
            "compare_path": _safe_rel(run_out / "compare_to_base.json", out_root) if compare else None,
            "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        answer_path = run_out / "gemini_answer.md"
        if args.gemini:
            old_manifest_path = run_out / "manifest.json"
            old_manifest = _load_json(old_manifest_path) if old_manifest_path.is_file() else {}
            can_reuse = (
                answer_path.is_file()
                and used_cache
                and not args.force_refresh
                and old_manifest.get("prompt_sha256") == prompt_sha256
                and old_manifest.get("report_verbosity") == args.report_verbosity
            )
            if can_reuse:
                manifest["gemini_answer_path"] = _safe_rel(answer_path, out_root)
                manifest["gemini_reused"] = True
            else:
                answer = _run_gemini(prompt, args.gemini_cmd, args.gemini_no_browser)
                _write_text(answer_path, answer + "\n")
                manifest["gemini_answer_path"] = _safe_rel(answer_path, out_root)
                manifest["gemini_reused"] = False

        _write_json(run_out / "manifest.json", manifest)
        print(f"[OK] {run_id}: snapshot/prompt saved under {run_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
