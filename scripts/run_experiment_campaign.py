from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], cwd: Path, timeout: int) -> Dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.time() - started, 3),
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "seconds": round(time.time() - started, 3),
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def _py(repo: Path, script: str, *args: str) -> List[str]:
    return [sys.executable, str(repo / "scripts" / script), *args]


def build_commands(args: argparse.Namespace, repo: Path, out: Path) -> List[Dict[str, Any]]:
    py = sys.executable
    commands = [
        {
            "name": "target_queue",
            "timeout": 120,
            "cmd": _py(
                repo,
                "plan_official_audit_targets.py",
                "--repo-root", str(repo),
                "--out", str(out / "official_audit_target_queue"),
                *(
                    ["--mlagentbench-root", args.mlagentbench_root]
                    if args.mlagentbench_root
                    else []
                ),
            ),
        },
        {
            "name": "build_public_failure_corpus",
            "timeout": 120,
            "cmd": _py(repo, "build_public_failure_corpus.py", "--repo-root", str(repo)),
        },
        {
            "name": "public_failure_corpus_single_seed",
            "timeout": 180,
            "cmd": _py(
                repo,
                "run_false_discovery_corpus_eval.py",
                "--repo-root", str(repo),
                "--out", str(out / "public_failure_corpus_eval"),
                "--traces-per-task", str(args.traces_per_task),
                "--max-candidates", str(args.max_candidates),
                "--seed", str(args.seed_start),
            ),
        },
        {
            "name": "public_failure_corpus_multiseed",
            "timeout": 600,
            "cmd": _py(
                repo,
                "run_public_corpus_multiseed_eval.py",
                "--repo-root", str(repo),
                "--out", str(out / "public_failure_corpus_multiseed_eval"),
                "--seeds", str(args.seeds),
                "--seed-start", str(args.seed_start),
                "--traces-per-task", str(args.traces_per_task),
                "--max-candidates", str(args.max_candidates),
            ),
        },
        {
            "name": "same_agent_policy_eval",
            "timeout": 180,
            "cmd": _py(repo, "run_same_agent_policy_eval.py", "--repo-root", str(repo), "--out", str(out / "same_agent_policy_eval")),
        },
        {
            "name": "external_audit_summary",
            "timeout": 120,
            "cmd": _py(repo, "summarize_external_audits.py", "--repo-root", str(repo), "--out", str(out / "external_audit_summary")),
        },
        {
            "name": "unit_tests",
            "timeout": 120,
            "cmd": [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        },
    ]
    if args.include_official and args.mlagentbench_root:
        official = [
            ("official_vectorization", "run_mlagentbench_vectorization_candidates.py", ["--repeats", "3", "--timeout", "90"]),
            ("official_cifar10", "run_mlagentbench_cifar10_candidates.py", ["--repeats", "1", "--timeout", "240"]),
            ("official_imdb", "run_mlagentbench_imdb_candidates.py", ["--repeats", "1", "--timeout", "240", "--hf-offline"]),
        ]
        if args.include_heavy:
            official.append(("official_clrs", "run_mlagentbench_clrs_candidates.py", ["--timeout", "900"]))
        for name, script, extra in official:
            commands.append({
                "name": name,
                "timeout": args.official_timeout,
                "cmd": _py(
                    repo,
                    script,
                    "--mlagentbench-root", args.mlagentbench_root,
                    "--out", str(out / name),
                    "--python", py,
                    *extra,
                ),
            })
    return commands


def render(results: List[Dict[str, Any]]) -> str:
    lines = ["# SciTriage Experiment Campaign\n"]
    lines.append("This campaign runs a batch of reproducible experiments without requiring interactive prompting.")
    lines.append("")
    lines.append("| Step | Return code | Seconds | Timed out |")
    lines.append("|---|---:|---:|---|")
    for item in results:
        lines.append(
            f"| `{item['name']}` | {item['returncode']} | {item['seconds']:.3f} | {item['timed_out']} |"
        )
    lines.append("")
    failed = [item for item in results if item["returncode"] != 0]
    if failed:
        lines.append("## Failed Or Blocked Steps\n")
        for item in failed:
            lines.append(f"### {item['name']}")
            lines.append("```text")
            lines.append((item.get("stderr") or item.get("stdout") or "").strip()[-3000:])
            lines.append("```")
            lines.append("")
    else:
        lines.append("All campaign steps completed successfully.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mlagentbench-root", default=None)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=20260522)
    parser.add_argument("--traces-per-task", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--include-official", action="store_true")
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--official-timeout", type=int, default=1800)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in build_commands(args, repo, out):
        result = _run(spec["cmd"], repo, spec["timeout"])
        result["name"] = spec["name"]
        (out / f"{spec['name']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    summary = {
        "repo_root": str(repo),
        "out": str(out),
        "results": results,
        "failed": [item["name"] for item in results if item["returncode"] != 0],
    }
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "CAMPAIGN_SUMMARY.md").write_text(render(results), encoding="utf-8")
    print(out / "campaign_summary.json")
    print(out / "CAMPAIGN_SUMMARY.md")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
