from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from textwrap import dedent
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scitriage.semantic_gate import select_with_semantic_gate


VARIANTS = {
    "step1_encoded_decoded": ["--train_steps=1", "--hint_mode=encoded_decoded"],
    "step1_decoded_only": ["--train_steps=1", "--hint_mode=decoded_only"],
    "step1_no_hints": ["--train_steps=1", "--hint_mode=none"],
    "invalid_no_checkpoint": None,
}


def _run_train(
    python: str,
    train_path: Path,
    work: Path,
    args: List[str],
    timeout: int,
    dataset_path: Path,
) -> Dict[str, object]:
    proc = subprocess.run(
        [
            python,
            str(train_path),
            *args,
            "--eval_every=1",
            "--test_every=1",
            "--log_every=1",
            "--checkpoint_path=./checkpoints",
            f"--dataset_path={dataset_path}",
        ],
        cwd=work,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _official_score(
    python: str,
    eval_path: Path,
    submission_dir: Path,
    timeout: int,
) -> Dict[str, object]:
    code = dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlagentbench_clrs_eval", {str(eval_path)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(module.get_score({str(submission_dir)!r}))
        """
    )
    try:
        proc = subprocess.run(
            [python, "-c", code],
            cwd=eval_path.parent,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "official_score": None,
            "stdout": "",
            "stderr": f"timeout_after_seconds: {timeout}",
        }
    score = None
    if proc.returncode == 0:
        try:
            score = float(proc.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            score = None
    return {
        "returncode": proc.returncode,
        "official_score": score,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _checkpoint_check(work: Path) -> Dict[str, object]:
    checkpoint = work / "checkpoints"
    required = ["best.pkl", "spec_list.pkl", "model_params.pkl"]
    present = {name: (checkpoint / name).exists() for name in required}
    return {
        "passed": all(present.values()),
        "present": present,
    }


def _render_markdown(result: Dict[str, object]) -> str:
    rows: List[Dict[str, object]] = result["rows"]
    scored = [r for r in rows if r["official_score"] is not None]
    visible = sorted(scored, key=lambda r: r["official_score"], reverse=True)
    valid = [r for r in scored if r["validity_gate_passed"]]
    triage = sorted(valid, key=lambda r: r["official_score"], reverse=True)
    lines = ["# MLAgentBench CLRS Candidate Audit\n"]
    lines.append("External task: `MLAgentBench/benchmarks/CLRS`.")
    lines.append("Official score is CLRS test score, higher is better.")
    lines.append("SciTriage checks checkpoint loadability via official eval.\n")
    if visible:
        lines.append(f"- Official best candidate: `{visible[0]['variant']}`")
    if triage:
        lines.append(f"- SciTriage-gated best candidate: `{triage[0]['variant']}`")
    lines.append(f"- Candidates blocked by validity gate: {sum(not r['validity_gate_passed'] for r in rows)}")
    lines.append("\n## Policy Comparison\n")
    lines.append("| Policy | Selected | Official Score | Validity Gate |")
    lines.append("|---|---|---:|---|")
    for name in ["visible_score_only", "scitriage_gated"]:
        item = result["policy_comparison"][name]
        score = "-" if item["official_score"] is None else f"{item['official_score']:.6f}"
        lines.append(f"| `{name}` | `{item['selected']}` | {score} | {item['validity_gate_passed']} |")
    lines.append("\n## Candidate Table\n")
    lines.append("| Variant | Official Score | Checkpoint | Official Eval | Triage Status |")
    lines.append("|---|---:|---|---|---|")
    for row in sorted(rows, key=lambda r: (r["official_score"] is None, -(r["official_score"] or -1.0))):
        score = "-" if row["official_score"] is None else f"{row['official_score']:.6f}"
        status = "allowed" if row["validity_gate_passed"] else "blocked"
        lines.append(
            f"| `{row['variant']}` | {score} | {row['checkpoint_passed']} | "
            f"{row['official_eval_passed']} | `{status}` |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "This audit expands SciTriage beyond CSV submissions. On CLRS, a candidate is valid only "
        "if it produces checkpoint artifacts that the official evaluator can load and score."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagentbench-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dataset-path", default="/tmp/CLRS30")
    args = parser.parse_args()

    root = Path(args.mlagentbench_root).expanduser().resolve()
    task = root / "MLAgentBench" / "benchmarks" / "CLRS"
    env_dir = task / "env"
    eval_path = task / "scripts" / "eval.py"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant, train_args in VARIANTS.items():
        work = out / "workspaces" / variant
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(env_dir, work)
        train = {"returncode": 0, "stdout": "", "stderr": "skipped intentionally"}
        if train_args is not None:
            train = _run_train(args.python, work / "train.py", work, train_args, args.timeout, dataset_path)
        checkpoint = _checkpoint_check(work)
        official = _official_score(args.python, eval_path, work, args.timeout) if checkpoint["passed"] else {
            "returncode": 1,
            "official_score": None,
            "stdout": "",
            "stderr": "checkpoint_missing",
        }
        row = {
            "variant": variant,
            "train_returncode": train["returncode"],
            "train_stdout": train["stdout"],
            "train_stderr": train["stderr"],
            "official_score": official["official_score"],
            "official_eval_returncode": official["returncode"],
            "official_eval_stdout": official["stdout"],
            "official_eval_stderr": official["stderr"],
            "checkpoint_passed": checkpoint["passed"],
            "checkpoint": checkpoint,
        }
        row["official_eval_passed"] = row["official_score"] is not None and row["official_eval_returncode"] == 0
        row["validity_gate_passed"] = bool(row["checkpoint_passed"] and row["official_eval_passed"])
        rows.append(row)
        (out / f"{variant}.json").write_text(json.dumps(row, indent=2))

    policy = select_with_semantic_gate(
        rows,
        score_key="official_score",
        invariant_key="validity_gate_passed",
        lower_is_better=False,
    )
    result = {
        "task": "MLAgentBench/CLRS",
        "rows": rows,
        "mean_valid_score": mean([r["official_score"] for r in rows if r["validity_gate_passed"]]),
        "policy_comparison": {
            "visible_score_only": {
                "selected": policy["visible_score_only"]["selected"],
                "official_score": policy["visible_score_only"]["score"],
                "validity_gate_passed": policy["visible_score_only"]["semantic_invariant_passed"],
            },
            "scitriage_gated": {
                "selected": policy["scitriage_gated"]["selected"] if policy["scitriage_gated"] else None,
                "official_score": policy["scitriage_gated"]["score"] if policy["scitriage_gated"] else None,
                "validity_gate_passed": (
                    policy["scitriage_gated"]["semantic_invariant_passed"]
                    if policy["scitriage_gated"]
                    else False
                ),
            },
        },
        "validity_gate": policy,
    }
    (out / "candidate_audit.json").write_text(json.dumps(result, indent=2))
    (out / "CANDIDATE_AUDIT.md").write_text(_render_markdown(result))
    print(out / "candidate_audit.json")
    print(out / "CANDIDATE_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
