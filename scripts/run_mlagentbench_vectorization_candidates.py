from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from statistics import mean, stdev
from typing import Dict, List

import numpy as np


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace_forward_body(text: str, body: str) -> str:
    start = text.index("        for index in range(batch_size):")
    end = text.index("        \n        \n        if self.activation == 'relu':", start)
    return text[:start] + body.rstrip() + "\n" + text[end:]


def _variant_text(base: str, variant: str) -> str:
    if variant == "baseline":
        return base
    if variant == "filter_vectorized":
        body = """
        for index in range(batch_size):
            padded_feature = padded_batch[index, :, :, :]
            for h in range(h_new):
                for w in range(w_new):
                    vertical_start = h * self.stride
                    vertical_end = vertical_start + filter_size
                    horizontal_start = w * self.stride
                    horizontal_end = horizontal_start + filter_size
                    image_portion = padded_feature[vertical_start:vertical_end, horizontal_start:horizontal_end, :]
                    output[index, h, w, :] = (
                        np.tensordot(image_portion, self.kernel_matrices, axes=([0, 1, 2], [0, 1, 2]))
                        + self.biases.reshape(-1)
                    )
        """
        return _replace_forward_body(base, body)
    if variant == "im2col_einsum":
        body = """
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(
            padded_batch,
            (filter_size, filter_size, num_features_old),
            axis=(1, 2, 3),
        )
        windows = windows[:, ::self.stride, ::self.stride, 0, :, :, :]
        output = np.tensordot(windows, self.kernel_matrices, axes=([3, 4, 5], [0, 1, 2]))
        output = output + self.biases.reshape(1, 1, 1, -1)
        """
        return _replace_forward_body(base, body)
    if variant == "zero_fast_invalid":
        body = """
        output = np.zeros([batch_size, h_new, w_new, num_of_filters_new])
        """
        return _replace_forward_body(base, body)
    if variant == "random_fast_invalid":
        body = """
        output = np.random.randn(batch_size, h_new, w_new, num_of_filters_new)
        """
        return _replace_forward_body(base, body)
    if variant == "bias_only_invalid":
        body = """
        output = np.broadcast_to(self.biases.reshape(1, 1, 1, -1), [batch_size, h_new, w_new, num_of_filters_new]).copy()
        """
        return _replace_forward_body(base, body)
    raise ValueError(f"unknown variant {variant}")


def _run_train(python: str, work_dir: Path, timeout: int) -> Dict[str, object]:
    submission = work_dir / "submission.csv"
    if submission.exists():
        submission.unlink()
    proc = subprocess.run(
        [python, "train.py"],
        cwd=work_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    score = None
    if submission.exists():
        with submission.open(newline="") as f:
            reader = csv.reader(f, delimiter=";")
            row = next(reader, None)
            if row:
                score = float(row[0])
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "official_score": score,
    }


def _run_repeated(python: str, work_dir: Path, timeout: int, repeats: int) -> Dict[str, object]:
    runs = [_run_train(python, work_dir, timeout) for _ in range(repeats)]
    scores = [r["official_score"] for r in runs if r["official_score"] is not None and r["returncode"] == 0]
    return {
        "runs": runs,
        "official_score": mean(scores) if scores else None,
        "official_score_std": stdev(scores) if len(scores) >= 2 else 0.0,
        "official_score_min": min(scores) if scores else None,
        "official_score_max": max(scores) if scores else None,
        "returncode": 0 if len(scores) == repeats else 1,
        "stdout": "\n".join(r["stdout"] for r in runs),
        "stderr": "\n".join(r["stderr"] for r in runs),
    }


def _semantic_check(original_path: Path, candidate_path: Path) -> Dict[str, object]:
    original = _load_module(original_path, f"orig_{candidate_path.parent.name}")
    candidate = _load_module(candidate_path, f"cand_{candidate_path.parent.name}")
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2, 9, 9, 3))
    orig_layer = original.Conv2DLayer(3, 4, 3, 2, 1, None)
    cand_layer = candidate.Conv2DLayer(3, 4, 3, 2, 1, None)
    cand_layer.kernel_matrices = orig_layer.kernel_matrices.copy()
    cand_layer.biases = orig_layer.biases.copy()
    y0 = orig_layer.forward(x)
    y1 = cand_layer.forward(x)
    max_abs_error = float(np.max(np.abs(y0 - y1)))
    return {
        "passed": bool(np.allclose(y0, y1, atol=1e-8, rtol=1e-8)),
        "max_abs_error": max_abs_error,
    }


def _write_log(path: Path, row: Dict[str, object]) -> None:
    path.write_text(dedent(f"""
    variant: {row['variant']}
    official_score: {row['official_score']}
    time_taken: {row['official_score']}
    returncode: {row['returncode']}
    semantic_invariant_passed: {row['semantic_invariant_passed']}
    max_abs_error: {row['max_abs_error']}
    stdout:
    {row['stdout']}
    stderr:
    {row['stderr']}
    """).strip() + "\n")


def _render_markdown(result: Dict[str, object]) -> str:
    rows: List[Dict[str, object]] = result["rows"]
    valid = [r for r in rows if r["semantic_invariant_passed"] and r["official_score"] is not None]
    official_sorted = sorted(
        [r for r in rows if r["official_score"] is not None],
        key=lambda r: r["official_score"],
    )
    triage_sorted = sorted(valid, key=lambda r: r["official_score"])
    lines = ["# MLAgentBench Vectorization Candidate Audit\n"]
    lines.append("External task: `MLAgentBench/benchmarks/vectorization`.")
    lines.append("Official score is runtime in seconds, lower is better.")
    lines.append("SciTriage adds a semantic invariant check against the original convolution output.\n")
    if official_sorted:
        lines.append(f"- Official best candidate: `{official_sorted[0]['variant']}`")
    if triage_sorted:
        lines.append(f"- SciTriage-gated best candidate: `{triage_sorted[0]['variant']}`")
    baseline = next((r for r in rows if r["variant"] == "baseline"), None)
    if baseline and triage_sorted and baseline["official_score"] and triage_sorted[0]["official_score"]:
        lines.append(
            "- SciTriage-gated speedup over baseline: "
            f"{baseline['official_score'] / triage_sorted[0]['official_score']:.1f}x"
        )
    blocked = [r for r in rows if not r["semantic_invariant_passed"]]
    lines.append(f"- Candidates blocked by semantic invariant: {len(blocked)}")
    lines.append(f"- Repeats per candidate: {result['repeats']}")
    lines.append("\n## Policy Comparison\n")
    policy = result["policy_comparison"]
    lines.append("| Policy | Selected | Official Score | Semantically Valid |")
    lines.append("|---|---|---:|---|")
    for name in ["visible_score_only", "scitriage_gated"]:
        item = policy[name]
        score = "-" if item["official_score"] is None else f"{item['official_score']:.6f}"
        lines.append(f"| `{name}` | `{item['selected']}` | {score} | {item['semantic_invariant_passed']} |")
    lines.append("\n## Candidate Table\n")
    lines.append("| Variant | Mean Official Score | Std | Invariant | Max Abs Error | Triage Status |")
    lines.append("|---|---:|---:|---|---:|---|")
    for row in sorted(rows, key=lambda r: (r["official_score"] is None, r["official_score"] or 1e99)):
        status = "allowed" if row["semantic_invariant_passed"] else "blocked"
        score = "-" if row["official_score"] is None else f"{row['official_score']:.6f}"
        std = "-" if row["official_score_std"] is None else f"{row['official_score_std']:.6f}"
        lines.append(
            f"| `{row['variant']}` | {score} | {std} | {row['semantic_invariant_passed']} | "
            f"{row['max_abs_error']:.6g} | `{status}` |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "This audit separates benchmark score from scientific validity. If an invalid shortcut "
        "wins the official runtime-only score, SciTriage blocks it and promotes the fastest "
        "candidate that preserves the original computation."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagentbench-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.mlagentbench_root).expanduser().resolve()
    env_dir = root / "MLAgentBench" / "benchmarks" / "vectorization" / "env"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_text = (env_dir / "train.py").read_text()
    rows = []
    variants = [
        "baseline",
        "filter_vectorized",
        "im2col_einsum",
        "zero_fast_invalid",
        "random_fast_invalid",
        "bias_only_invalid",
    ]
    for variant in variants:
        work = out / "workspaces" / variant
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(env_dir, work)
        train_path = work / "train.py"
        train_path.write_text(_variant_text(base_text, variant))
        run = _run_repeated(args.python, work, args.timeout, args.repeats)
        semantic = _semantic_check(env_dir / "train.py", train_path)
        row = {
            "variant": variant,
            **run,
            "semantic_invariant_passed": semantic["passed"],
            "max_abs_error": semantic["max_abs_error"],
        }
        rows.append(row)
        _write_log(out / f"{variant}.log", row)

    official = min((r for r in rows if r["official_score"] is not None), key=lambda r: r["official_score"])
    valid = [r for r in rows if r["official_score"] is not None and r["semantic_invariant_passed"]]
    triage = min(valid, key=lambda r: r["official_score"])
    result = {
        "task": "MLAgentBench/vectorization",
        "repeats": args.repeats,
        "rows": rows,
        "policy_comparison": {
            "visible_score_only": {
                "selected": official["variant"],
                "official_score": official["official_score"],
                "semantic_invariant_passed": official["semantic_invariant_passed"],
            },
            "scitriage_gated": {
                "selected": triage["variant"],
                "official_score": triage["official_score"],
                "semantic_invariant_passed": triage["semantic_invariant_passed"],
            },
        },
    }
    (out / "candidate_audit.json").write_text(json.dumps(result, indent=2))
    (out / "CANDIDATE_AUDIT.md").write_text(_render_markdown(result))
    print(out / "candidate_audit.json")
    print(out / "CANDIDATE_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
