from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from textwrap import dedent
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scitriage.semantic_gate import select_with_semantic_gate


CLASSES = [0, 1]
TEST_SIZE = 25_000


def _candidate_source(variant: str) -> str:
    if variant == "uniform_valid":
        return dedent(
            """
            import numpy as np
            import pandas as pd

            rows = np.ones((25000, 2), dtype=float) / 2.0
            pd.DataFrame(rows, columns=[0, 1], index=range(25000)).to_csv("submission.csv", index_label="idx")
            """
        ).strip() + "\n"
    if variant == "random_valid":
        return dedent(
            """
            import numpy as np
            import pandas as pd

            rng = np.random.default_rng(123)
            rows = rng.random((25000, 2))
            rows = rows / rows.sum(axis=1, keepdims=True)
            pd.DataFrame(rows, columns=[0, 1], index=range(25000)).to_csv("submission.csv", index_label="idx")
            """
        ).strip() + "\n"
    if variant == "train_prior_valid":
        return dedent(
            """
            import numpy as np
            import pandas as pd
            from datasets import load_dataset

            imdb = load_dataset("imdb")
            counts = np.zeros(2, dtype=float)
            for row in imdb["train"]:
                counts[row["label"]] += 1.0
            probs = counts / counts.sum()
            rows = np.tile(probs.reshape(1, 2), (25000, 1))
            pd.DataFrame(rows, columns=[0, 1], index=range(25000)).to_csv("submission.csv", index_label="idx")
            """
        ).strip() + "\n"
    if variant == "test_label_oracle_invalid":
        return dedent(
            """
            import numpy as np
            import pandas as pd
            from datasets import load_dataset

            imdb = load_dataset("imdb")
            rows = np.zeros((len(imdb["test"]), 2), dtype=float)
            for idx, row in enumerate(imdb["test"]):
                rows[idx, row["label"]] = 1.0
            pd.DataFrame(rows, columns=[0, 1], index=range(len(imdb["test"]))).to_csv("submission.csv", index_label="idx")
            """
        ).strip() + "\n"
    if variant == "test_label_flipped_invalid":
        return dedent(
            """
            import numpy as np
            import pandas as pd
            from datasets import load_dataset

            imdb = load_dataset("imdb")
            rows = np.zeros((len(imdb["test"]), 2), dtype=float)
            for idx, row in enumerate(imdb["test"]):
                rows[idx, 1 - row["label"]] = 1.0
            pd.DataFrame(rows, columns=[0, 1], index=range(len(imdb["test"]))).to_csv("submission.csv", index_label="idx")
            """
        ).strip() + "\n"
    raise ValueError(f"unknown variant {variant}")


class _LoadDatasetVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.dataset_vars: set[str] = set()
        self.leaks: List[Dict[str, object]] = []

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if _is_imdb_load_dataset(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.dataset_vars.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        split = _iter_dataset_split(node.iter)
        if split and split["dataset"] in self.dataset_vars and split["split"] == "test":
            loaded_names = _load_names(node.body)
            row_names = _row_names_from_target(node.target)
            leak = _body_reads_label(node.body, row_names)
            if leak:
                self.leaks.append({
                    "line": node.lineno,
                    "dataset": split["dataset"],
                    "split": "test",
                    "row_names": sorted(row_names),
                    "loaded_names": sorted(loaded_names),
                })
        self.generic_visit(node)


class _LoadNameFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _is_imdb_load_dataset(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_load_dataset = (
        isinstance(func, ast.Name) and func.id == "load_dataset"
    ) or (
        isinstance(func, ast.Attribute) and func.attr == "load_dataset"
    )
    if not is_load_dataset or not node.args:
        return False
    return isinstance(node.args[0], ast.Constant) and node.args[0].value == "imdb"


def _iter_dataset_split(node: ast.AST) -> Dict[str, str] | None:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and _slice_value(node.slice) == "test"
    ):
        return {"dataset": node.value.id, "split": "test"}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
        and node.args
    ):
        return _iter_dataset_split(node.args[0])
    return None


def _slice_value(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _row_names_from_target(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple):
        names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        return set(names[-1:])
    return set()


def _body_reads_label(nodes: Iterable[ast.AST], row_names: set[str]) -> bool:
    for node in nodes:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id in row_names
                and _slice_value(child.slice) == "label"
            ):
                return True
    return False


def _load_names(nodes: Iterable[ast.AST]) -> set[str]:
    finder = _LoadNameFinder()
    for node in nodes:
        finder.visit(node)
    return finder.names


def audit_test_label_leak(source: str) -> Dict[str, object]:
    tree = ast.parse(source)
    visitor = _LoadDatasetVisitor()
    visitor.visit(tree)
    return {
        "passed": len(visitor.leaks) == 0,
        "test_label_leaks": visitor.leaks,
        "dataset_vars": sorted(visitor.dataset_vars),
    }


def _run_candidate(python: str, work_dir: Path, timeout: int, env: Dict[str, str]) -> Dict[str, object]:
    submission = work_dir / "submission.csv"
    if submission.exists():
        submission.unlink()
    try:
        proc = subprocess.run(
            [python, "train.py"],
            cwd=work_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\ntimeout_after_seconds: {timeout}",
            "submission_exists": submission.exists(),
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "submission_exists": submission.exists(),
    }


def _official_score(
    python: str,
    eval_path: Path,
    submission_dir: Path,
    timeout: int,
    env: Dict[str, str],
) -> float | None:
    code = dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlagentbench_imdb_eval", {str(eval_path)!r})
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
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def _schema_check(path: Path) -> Dict[str, object]:
    submission = path / "submission.csv"
    if not submission.exists():
        return {"passed": False, "reason": "missing_submission"}
    try:
        df = pd.read_csv(submission, index_col="idx")
    except Exception as exc:  # pragma: no cover - defensive for external artifacts
        return {"passed": False, "reason": f"unreadable_submission:{exc}"}
    numeric = df.apply(pd.to_numeric, errors="coerce")
    row_sums = numeric.sum(axis=1)
    return {
        "passed": (
            df.shape == (TEST_SIZE, 2)
            and list(map(str, df.columns)) == list(map(str, CLASSES))
            and numeric.notna().all().all()
            and bool(np.allclose(row_sums.to_numpy(dtype=float), 1.0, atol=1e-5))
        ),
        "shape": list(df.shape),
        "columns": list(map(str, df.columns)),
        "min_row_sum": float(row_sums.min()) if len(row_sums) else None,
        "max_row_sum": float(row_sums.max()) if len(row_sums) else None,
    }


def _run_repeated(
    python: str,
    eval_path: Path,
    work_dir: Path,
    timeout: int,
    repeats: int,
    env: Dict[str, str],
) -> Dict[str, object]:
    runs = []
    for _ in range(repeats):
        run = _run_candidate(python, work_dir, timeout, env)
        run["official_score"] = _official_score(python, eval_path, work_dir, timeout, env) if run["submission_exists"] else None
        run["schema_check"] = _schema_check(work_dir)
        runs.append(run)
    scores = [r["official_score"] for r in runs if r["returncode"] == 0 and r["official_score"] is not None]
    return {
        "runs": runs,
        "official_score": mean(scores) if scores else None,
        "official_score_std": stdev(scores) if len(scores) >= 2 else 0.0,
        "official_score_min": min(scores) if scores else None,
        "official_score_max": max(scores) if scores else None,
        "returncode": 0 if len(scores) == repeats else 1,
        "stdout": "\n".join(str(r["stdout"]) for r in runs),
        "stderr": "\n".join(str(r["stderr"]) for r in runs),
        "schema_passed": all(bool(r["schema_check"]["passed"]) for r in runs),
    }


def _render_markdown(result: Dict[str, object]) -> str:
    rows: List[Dict[str, object]] = result["rows"]
    scored = [r for r in rows if r["official_score"] is not None]
    official_sorted = sorted(scored, key=lambda r: r["official_score"], reverse=True)
    valid = [r for r in scored if r["validity_gate_passed"]]
    triage_sorted = sorted(valid, key=lambda r: r["official_score"], reverse=True)
    lines = ["# MLAgentBench IMDB Candidate Audit\n"]
    lines.append("External task: `MLAgentBench/benchmarks/imdb`.")
    lines.append("Official score is IMDB test accuracy, higher is better.")
    lines.append("SciTriage adds a test-label-leak gate and a submission-schema check.\n")
    if official_sorted:
        lines.append(f"- Official best candidate: `{official_sorted[0]['variant']}`")
    if triage_sorted:
        lines.append(f"- SciTriage-gated best candidate: `{triage_sorted[0]['variant']}`")
    blocked = [r for r in rows if not r["validity_gate_passed"]]
    lines.append(f"- Candidates blocked by validity gate: {len(blocked)}")
    lines.append(f"- Repeats per candidate: {result['repeats']}")
    lines.append("\n## Policy Comparison\n")
    policy = result["policy_comparison"]
    lines.append("| Policy | Selected | Official Accuracy | Validity Gate |")
    lines.append("|---|---|---:|---|")
    for name in ["visible_score_only", "scitriage_gated"]:
        item = policy[name]
        score = "-" if item["official_score"] is None else f"{item['official_score']:.4f}"
        lines.append(f"| `{name}` | `{item['selected']}` | {score} | {item['validity_gate_passed']} |")
    lines.append("\n## Candidate Table\n")
    lines.append("| Variant | Mean Official Accuracy | Std | Schema | Test-Label Leak Gate | Triage Status |")
    lines.append("|---|---:|---:|---|---|---|")
    for row in sorted(rows, key=lambda r: (r["official_score"] is None, -(r["official_score"] or -1.0))):
        status = "allowed" if row["validity_gate_passed"] else "blocked"
        score = "-" if row["official_score"] is None else f"{row['official_score']:.4f}"
        std = "-" if row["official_score_std"] is None else f"{row['official_score_std']:.4f}"
        lines.append(
            f"| `{row['variant']}` | {score} | {std} | {row['schema_passed']} | "
            f"{row['test_label_gate_passed']} | `{status}` |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "This audit repeats the CIFAR-10 leakage story on a different MLAgentBench task. "
        "A candidate can read IMDB test labels and win the official accuracy metric without "
        "learning sentiment classification. SciTriage blocks that candidate before it becomes "
        "a false AutoResearch success."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagentbench-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hf-endpoint", default=None)
    args = parser.parse_args()

    root = Path(args.mlagentbench_root).expanduser().resolve()
    task_dir = root / "MLAgentBench" / "benchmarks" / "imdb"
    eval_path = task_dir / "scripts" / "eval.py"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    variants = [
        "uniform_valid",
        "random_valid",
        "train_prior_valid",
        "test_label_flipped_invalid",
        "test_label_oracle_invalid",
    ]
    rows = []
    for variant in variants:
        work = out / "workspaces" / variant
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        source = _candidate_source(variant)
        (work / "train.py").write_text(source)
        leak = audit_test_label_leak(source)
        run = _run_repeated(args.python, eval_path, work, args.timeout, args.repeats, env)
        validity = bool(leak["passed"] and run["schema_passed"] and run["returncode"] == 0)
        row = {
            "variant": variant,
            **run,
            "test_label_gate_passed": leak["passed"],
            "test_label_gate": leak,
            "validity_gate_passed": validity,
        }
        rows.append(row)
        (out / f"{variant}.json").write_text(json.dumps(row, indent=2))

    policy = select_with_semantic_gate(
        rows,
        score_key="official_score",
        invariant_key="validity_gate_passed",
        lower_is_better=False,
    )
    result = {
        "task": "MLAgentBench/imdb",
        "repeats": args.repeats,
        "rows": rows,
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
