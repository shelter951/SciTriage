from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Dict


AUTORESEARCH_HOOK = """\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scitriage.plugin import (
    assess_autoresearch_run,
    prioritize_candidate,
    resource_fit,
    seed_group_gate,
    write_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SciTriage after an AutoResearch candidate.")
    parser.add_argument("--repo", default=".", help="AutoResearch repository or candidate workspace")
    parser.add_argument("--run-log", required=True, help="Candidate run log")
    parser.add_argument("--out", default="runs/scitriage_latest", help="Output directory")
    parser.add_argument("--baseline-val-bpb", type=float, help="Baseline mean val_bpb for one-run triage")
    parser.add_argument("--baseline-std", type=float, help="Observed baseline seed std")
    parser.add_argument("--baseline-summary", help="Seed summary JSON used for probe priority")
    parser.add_argument("--baseline-logs", nargs="*", default=[], help="Baseline seed logs")
    parser.add_argument("--candidate-logs", nargs="*", default=[], help="Candidate seed logs")
    parser.add_argument("--metric", default="val_bpb")
    parser.add_argument("--claim", action="append", default=[])
    parser.add_argument("--proposal", default="")
    parser.add_argument("--diff-ref")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bundle = {"resource_fit": resource_fit(args.run_log)}
    if args.baseline_val_bpb is not None:
        bundle.update(assess_autoresearch_run(
            repo=args.repo,
            run_log=args.run_log,
            baseline_val_bpb=args.baseline_val_bpb,
            baseline_std=args.baseline_std,
            claims=args.claim,
            proposal=args.proposal,
            diff_ref=args.diff_ref,
        ))
    if args.baseline_summary:
        bundle["probe_priority"] = prioritize_candidate(args.run_log, args.baseline_summary, args.metric)
    if args.baseline_logs and args.candidate_logs:
        claim = args.claim[0] if args.claim else "Candidate improves the target metric."
        bundle["seed_group_gate"] = seed_group_gate(
            baseline_logs=args.baseline_logs,
            candidate_logs=args.candidate_logs,
            metric=args.metric,
            claim=claim,
        )

    paths = write_bundle(bundle, out)
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def write_autoresearch_hook(out: str | Path) -> Dict[str, str]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(AUTORESEARCH_HOOK))
    try:
        path.chmod(0o755)
    except OSError:
        pass
    return {
        "hook_path": str(path),
        "purpose": "post-run SciTriage hook for Karpathy-style AutoResearch workspaces",
    }
