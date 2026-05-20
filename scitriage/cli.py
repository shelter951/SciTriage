from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import render_markdown
from .rules import diagnose
from .schema import ResearchTrace
from .adapters.autoresearch import trace_from_autoresearch
from .adapters.filesystem import trace_from_filesystem
from .benchmark import evaluate_benchmark, render_benchmark_markdown
from .probe_plan import build_probe_plan, render_probe_plan_markdown
from .aggregate import aggregate_logs, compare_seed_groups
from .toy_eval import run_toy_policy_eval, run_toy_policy_sweep, render_toy_policy_markdown
from .toy_materialize import materialize_toy_cases
from .real_summary import summarize_real_runs, render_real_summary_markdown
from .claim_rewrite import reconstruct_claims, render_reconstructed_claims
from .resource_fit import diagnose_resource_fit, render_resource_fit
from .autoresearch_probe import materialize_autoresearch_probe, render_probe_manifest
from .discovery_audit import audit_discoveries, render_discovery_audit_markdown
from .claim_gate import gate_claim, render_claim_gate_markdown
from .probe_priority import prioritize_probe, render_probe_priority_markdown
from .candidate_manifest import (
    build_candidate_manifest,
    write_candidate_manifest,
    summarize_candidate_manifests,
    render_candidate_summary_markdown,
)


def cmd_assess(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace_json)
    trace = ResearchTrace.from_dict(json.loads(trace_path.read_text()))
    report = diagnose(trace)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "triage_report.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        (out / "triage_report.md").write_text(render_markdown(report))
        print(out / "triage_report.json")
        print(out / "triage_report.md")
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0



def cmd_assess_filesystem(args: argparse.Namespace) -> int:
    trace = trace_from_filesystem(args.root, trace_id=args.trace_id)
    report = diagnose(trace)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'trace.json').write_text(json.dumps({
            'trace_id': trace.trace_id,
            'question': trace.question,
            'proposal': trace.proposal,
            'claims': trace.claims,
            'changed_files': trace.changed_files,
            'protected_files': trace.protected_files,
            'diff_summary': trace.diff_summary,
            'logs': trace.logs,
            'metrics': [m.__dict__ for m in trace.metrics],
            'experiment': trace.experiment,
            'history': trace.history,
        }, indent=2, ensure_ascii=False))
        (out / 'triage_report.json').write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        (out / 'triage_report.md').write_text(render_markdown(report))
        from .probe_plan import build_probe_plan, render_probe_plan_markdown
        plan = build_probe_plan(trace, report)
        (out / 'probe_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        (out / 'probe_plan.md').write_text(render_probe_plan_markdown(plan))
        print(out / 'trace.json')
        print(out / 'triage_report.md')
        print(out / 'probe_plan.md')
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_assess_autoresearch(args: argparse.Namespace) -> int:
    trace = trace_from_autoresearch(
        repo=args.repo,
        run_log=args.run_log,
        baseline_val_bpb=args.baseline_val_bpb,
        baseline_std=args.baseline_std,
        seeds=args.seeds,
        claims=args.claim,
        proposal=args.proposal or "",
        diff_ref=args.diff_ref,
    )
    report = diagnose(trace)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.json").write_text(json.dumps({
            "trace_id": trace.trace_id,
            "question": trace.question,
            "proposal": trace.proposal,
            "claims": trace.claims,
            "changed_files": trace.changed_files,
            "protected_files": trace.protected_files,
            "diff_summary": trace.diff_summary,
            "logs": trace.logs,
            "metrics": [m.__dict__ for m in trace.metrics],
            "experiment": trace.experiment,
            "history": trace.history,
        }, indent=2, ensure_ascii=False))
        (out / "triage_report.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        (out / "triage_report.md").write_text(render_markdown(report))
        print(out / "trace.json")
        print(out / "triage_report.json")
        print(out / "triage_report.md")
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_eval_benchmark(args: argparse.Namespace) -> int:
    summary = evaluate_benchmark(args.path)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (out / "benchmark_report.md").write_text(render_benchmark_markdown(summary))
        print(out / "benchmark_report.json")
        print(out / "benchmark_report.md")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_plan_probe(args: argparse.Namespace) -> int:
    trace = ResearchTrace.from_dict(json.loads(Path(args.trace_json).read_text()))
    report_data = json.loads(Path(args.report_json).read_text())
    from .schema import ProbeRecommendation, TriageReport
    p = report_data.get('recommended_probe')
    probe = None if p is None else ProbeRecommendation(
        probe_type=p.get('type'),
        reason=p.get('reason', ''),
        estimated_cost_gpu_hours=p.get('estimated_cost_gpu_hours', 0.0),
        details=p.get('details', {}),
    )
    report = TriageReport(
        trace_id=report_data.get('trace_id', trace.trace_id),
        status=report_data.get('status', 'unknown'),
        risk_labels=report_data.get('risk_labels', []),
        evidence_debt=report_data.get('evidence_debt', []),
        recommended_probe=probe,
        allowed_claims=report_data.get('allowed_claims', []),
        blocked_claims=report_data.get('blocked_claims', []),
        rationales=report_data.get('rationales', []),
    )
    plan = build_probe_plan(trace, report)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'probe_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        (out / 'probe_plan.md').write_text(render_probe_plan_markdown(plan))
        print(out / 'probe_plan.json')
        print(out / 'probe_plan.md')
    else:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cmd_aggregate_seeds(args: argparse.Namespace) -> int:
    summary = aggregate_logs(args.logs, metric=args.metric, baseline=args.baseline, higher_is_better=args.higher_is_better)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(out)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_compare_seed_groups(args: argparse.Namespace) -> int:
    summary = compare_seed_groups(
        args.baseline_logs,
        args.candidate_logs,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
        z=args.z,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(out)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_toy_policy_eval(args: argparse.Namespace) -> int:
    result = run_toy_policy_eval()
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'toy_policy_eval.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'toy_policy_eval.md').write_text(render_toy_policy_markdown(result))
        print(out / 'toy_policy_eval.json')
        print(out / 'toy_policy_eval.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_toy_policy_sweep(args: argparse.Namespace) -> int:
    result = run_toy_policy_sweep(args.n_per_type)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'toy_policy_sweep.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'toy_policy_sweep.md').write_text(render_toy_policy_markdown(result))
        print(out / 'toy_policy_sweep.json')
        print(out / 'toy_policy_sweep.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_toy_materialize(args: argparse.Namespace) -> int:
    summary = materialize_toy_cases(args.out)
    print(Path(args.out) / 'index.json')
    print(Path(args.out) / 'index.md')
    print(json.dumps({'num_cases': summary['num_cases']}, indent=2))
    return 0


def cmd_summarize_real(args: argparse.Namespace) -> int:
    summary = summarize_real_runs(args.run_dirs)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'real_artifact_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (out / 'real_artifact_summary.md').write_text(render_real_summary_markdown(summary))
        print(out / 'real_artifact_summary.json')
        print(out / 'real_artifact_summary.md')
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_reconstruct_claims(args: argparse.Namespace) -> int:
    result = reconstruct_claims(args.trace_json, args.report_json)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'reconstructed_claims.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'SUPPORTED_CLAIMS.md').write_text(render_reconstructed_claims(result))
        print(out / 'reconstructed_claims.json')
        print(out / 'SUPPORTED_CLAIMS.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_resource_fit(args: argparse.Namespace) -> int:
    result = diagnose_resource_fit(args.run_log)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'resource_fit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'resource_fit.md').write_text(render_resource_fit(result))
        print(out / 'resource_fit.json')
        print(out / 'resource_fit.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_materialize_autoresearch_probe(args: argparse.Namespace) -> int:
    manifest = materialize_autoresearch_probe(args.source_repo, args.target_dir)
    target = Path(args.target_dir).expanduser().resolve()
    (target / 'SCITRIAGE_PROBE_MANIFEST.md').write_text(render_probe_manifest(manifest))
    print(target / 'SCITRIAGE_PROBE_MANIFEST.json')
    print(target / 'SCITRIAGE_PROBE_MANIFEST.md')
    return 0


def cmd_audit_discoveries(args: argparse.Namespace) -> int:
    audit = audit_discoveries(args.seed1_sweep, args.group_compares)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'discovery_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        (out / 'discovery_audit.md').write_text(render_discovery_audit_markdown(audit))
        print(out / 'discovery_audit.json')
        print(out / 'discovery_audit.md')
    else:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


def cmd_claim_gate(args: argparse.Namespace) -> int:
    result = gate_claim(args.group_compare, args.claim, min_margin_ratio=args.min_margin_ratio)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'claim_gate.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'claim_gate.md').write_text(render_claim_gate_markdown(result))
        print(out / 'claim_gate.json')
        print(out / 'claim_gate.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_prioritize_probe(args: argparse.Namespace) -> int:
    result = prioritize_probe(
        args.candidate_log,
        args.baseline_summary,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
    )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'probe_priority.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        (out / 'probe_priority.md').write_text(render_probe_priority_markdown(result))
        print(out / 'probe_priority.json')
        print(out / 'probe_priority.md')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_write_candidate_manifest(args: argparse.Namespace) -> int:
    manifest = build_candidate_manifest(
        variant_id=args.variant_id,
        family=args.family,
        workspace=args.workspace,
        base_variant=args.base_variant,
        patch_intent=args.patch_intent,
        claim=args.claim,
        changed_files=args.changed_file,
        seed_logs=args.seed_log,
        notes=args.notes or "",
    )
    write_candidate_manifest(manifest, args.out)
    print(args.out)
    return 0


def cmd_summarize_candidate_manifests(args: argparse.Namespace) -> int:
    summary = summarize_candidate_manifests(args.manifests)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'candidate_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (out / 'candidate_summary.md').write_text(render_candidate_summary_markdown(summary))
        print(out / 'candidate_summary.json')
        print(out / 'candidate_summary.md')
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scitriage")
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser("assess", help="Assess one generic AutoResearch trace JSON")
    assess.add_argument("trace_json")
    assess.add_argument("--out")
    assess.set_defaults(func=cmd_assess)

    fs = sub.add_parser("assess-filesystem", help="Assess a real AutoResearch artifact directory")
    fs.add_argument("--root", required=True)
    fs.add_argument("--trace-id")
    fs.add_argument("--out")
    fs.set_defaults(func=cmd_assess_filesystem)

    ar = sub.add_parser("assess-autoresearch", help="Assess a Karpathy autoresearch run.log")
    ar.add_argument("--repo", required=True)
    ar.add_argument("--run-log", required=True)
    ar.add_argument("--baseline-val-bpb", type=float, required=True)
    ar.add_argument("--baseline-std", type=float)
    ar.add_argument("--seeds", type=int, default=1)
    ar.add_argument("--claim", action="append", default=[])
    ar.add_argument("--proposal")
    ar.add_argument("--diff-ref")
    ar.add_argument("--out")
    ar.set_defaults(func=cmd_assess_autoresearch)

    bench = sub.add_parser("eval-benchmark", help="Evaluate SciTriage on failure-injected traces")
    bench.add_argument("path")
    bench.add_argument("--out")
    bench.set_defaults(func=cmd_eval_benchmark)

    plan = sub.add_parser("plan-probe", help="Create command templates for the recommended diagnostic probe")
    plan.add_argument("trace_json")
    plan.add_argument("report_json")
    plan.add_argument("--out")
    plan.set_defaults(func=cmd_plan_probe)

    agg = sub.add_parser("aggregate-seeds", help="Aggregate metric values across seed logs")
    agg.add_argument("logs", nargs="+")
    agg.add_argument("--metric", required=True)
    agg.add_argument("--baseline", type=float)
    agg.add_argument("--higher-is-better", action="store_true")
    agg.add_argument("--out")
    agg.set_defaults(func=cmd_aggregate_seeds)

    cmp = sub.add_parser("compare-seed-groups", help="Compare baseline and candidate seed groups with a noise margin")
    cmp.add_argument("--baseline-logs", nargs="+", required=True)
    cmp.add_argument("--candidate-logs", nargs="+", required=True)
    cmp.add_argument("--metric", required=True)
    cmp.add_argument("--higher-is-better", action="store_true")
    cmp.add_argument("--z", type=float, default=1.96)
    cmp.add_argument("--out")
    cmp.set_defaults(func=cmd_compare_seed_groups)

    toy = sub.add_parser("toy-policy-eval", help="Run executable toy policy comparison")
    toy.add_argument("--out")
    toy.set_defaults(func=cmd_toy_policy_eval)

    sweep = sub.add_parser("toy-policy-sweep", help="Run deterministic generated toy policy sweep")
    sweep.add_argument("--n-per-type", type=int, default=20)
    sweep.add_argument("--out")
    sweep.set_defaults(func=cmd_toy_policy_sweep)

    materialize = sub.add_parser("toy-materialize", help="Write trace/report/probe artifacts for toy scenarios")
    materialize.add_argument("--out", required=True)
    materialize.set_defaults(func=cmd_toy_materialize)

    real = sub.add_parser("summarize-real", help="Summarize multiple real-artifact SciTriage outputs")
    real.add_argument("run_dirs", nargs="+")
    real.add_argument("--out")
    real.set_defaults(func=cmd_summarize_real)

    rc = sub.add_parser("reconstruct-claims", help="Rewrite blocked claims into evidence-safe statements")
    rc.add_argument("trace_json")
    rc.add_argument("report_json")
    rc.add_argument("--out")
    rc.set_defaults(func=cmd_reconstruct_claims)

    fit = sub.add_parser("resource-fit", help="Diagnose resource-fit failures such as autoresearch OOM")
    fit.add_argument("--run-log", required=True)
    fit.add_argument("--out")
    fit.set_defaults(func=cmd_resource_fit)

    mat = sub.add_parser("materialize-autoresearch-probe", help="Create a 4090-friendly autoresearch probe copy")
    mat.add_argument("--source-repo", required=True)
    mat.add_argument("--target-dir", required=True)
    mat.set_defaults(func=cmd_materialize_autoresearch_probe)

    audit = sub.add_parser("audit-discoveries", help="Audit one-shot AutoResearch discoveries against seed-group evidence")
    audit.add_argument("--seed1-sweep", required=True)
    audit.add_argument("--group-compares", nargs="+", required=True)
    audit.add_argument("--out")
    audit.set_defaults(func=cmd_audit_discoveries)

    gate = sub.add_parser("claim-gate", help="Allow or block a claim using seed-group evidence")
    gate.add_argument("--group-compare", required=True)
    gate.add_argument("--claim", required=True)
    gate.add_argument("--min-margin-ratio", type=float, default=1.0)
    gate.add_argument("--out")
    gate.set_defaults(func=cmd_claim_gate)

    priority = sub.add_parser("prioritize-probe", help="Prioritize a candidate probe using one-shot delta and measured baseline noise")
    priority.add_argument("--candidate-log", required=True)
    priority.add_argument("--baseline-summary", required=True)
    priority.add_argument("--metric", required=True)
    priority.add_argument("--higher-is-better", action="store_true")
    priority.add_argument("--out")
    priority.set_defaults(func=cmd_prioritize_probe)

    manifest = sub.add_parser("write-candidate-manifest", help="Write a reproducible manifest for one candidate variant")
    manifest.add_argument("--variant-id", required=True)
    manifest.add_argument("--family", required=True)
    manifest.add_argument("--workspace", required=True)
    manifest.add_argument("--base-variant", required=True)
    manifest.add_argument("--patch-intent", required=True)
    manifest.add_argument("--claim", required=True)
    manifest.add_argument("--changed-file", action="append", default=[])
    manifest.add_argument("--seed-log", action="append", default=[])
    manifest.add_argument("--notes")
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=cmd_write_candidate_manifest)

    manifest_summary = sub.add_parser("summarize-candidate-manifests", help="Summarize candidate manifests for an experiment family")
    manifest_summary.add_argument("manifests", nargs="+")
    manifest_summary.add_argument("--out")
    manifest_summary.set_defaults(func=cmd_summarize_candidate_manifests)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
