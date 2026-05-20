from __future__ import annotations

from typing import Dict, List

from .schema import ResearchTrace, TriageReport


def build_probe_plan(trace: ResearchTrace, report: TriageReport) -> Dict[str, object]:
    probe = report.recommended_probe
    if probe is None:
        return {
            'trace_id': trace.trace_id,
            'status': 'no_probe_needed',
            'commands': [],
            'notes': ['No diagnostic probe is recommended for this trace.'],
        }

    commands: List[Dict[str, object]] = []
    notes: List[str] = []
    ptype = probe.probe_type

    if ptype == 'multi_seed_rerun':
        seeds = probe.details.get('seeds', [1, 2, 3])
        notes.append('Run the same candidate and baseline under multiple seeds before accepting any performance claim.')
        for seed in seeds:
            commands.append({
                'name': f'candidate_seed_{seed}',
                'command_template': f'SEED={seed} <run-candidate-command> > runs/seed_{seed}.log 2>&1',
                'expected_artifact': f'runs/seed_{seed}.log',
            })
        commands.append({
            'name': 'aggregate_seed_metrics',
            'command_template': 'scitriage aggregate-seeds runs/seed_*.log --metric <metric-name>',
            'expected_artifact': 'runs/seed_summary.json',
        })

    elif ptype == 'frozen_evaluator':
        notes.append('Evaluation/protected files changed. Re-evaluate with trusted evaluator files restored or mounted read-only.')
        commands.append({
            'name': 'list_evaluator_diff',
            'command_template': 'git diff -- eval.py metrics.py prepare.py',
            'expected_artifact': 'evaluator_diff.patch',
        })
        commands.append({
            'name': 'rerun_with_frozen_evaluator',
            'command_template': '<restore-or-mount-frozen-evaluator> && <run-candidate-command>',
            'expected_artifact': 'runs/frozen_evaluator.log',
        })

    elif ptype == 'implementation_invariant':
        notes.append('Check implementation invariants before treating the failed run as evidence against the idea.')
        commands.extend([
            {'name': 'syntax_check', 'command_template': 'python3 -m compileall -q .', 'expected_artifact': 'compile_status'},
            {'name': 'smoke_test', 'command_template': '<short-smoke-test-command>', 'expected_artifact': 'runs/smoke_test.log'},
            {'name': 'finite_loss_check', 'command_template': '<run-one-batch-and-check-finite-loss>', 'expected_artifact': 'runs/finite_loss.json'},
        ])

    elif ptype == 'minimal_ablation':
        notes.append('Patch is confounded. Create minimal variants to isolate the claimed mechanism.')
        for idx, file in enumerate(trace.changed_files, 1):
            commands.append({
                'name': f'ablate_hunk_group_{idx}',
                'command_template': f'<create-variant-keeping-only-or-dropping-changes-in {file}> && <run-candidate-command>',
                'expected_artifact': f'runs/ablation_{idx}.log',
            })

    elif ptype == 'claim_reconstruction':
        notes.append('Rewrite claims from raw logs and metrics, removing unsupported causal/generalization/efficiency wording.')
        commands.append({
            'name': 'reconstruct_claims',
            'command_template': 'scitriage reconstruct-claims <trace.json> <triage_report.json>',
            'expected_artifact': 'SUPPORTED_CLAIMS.md',
        })

    elif ptype == 'budget_extension':
        notes.append('Pilot is too short. Extend budget before drawing conclusions.')
        commands.append({
            'name': 'extended_budget_run',
            'command_template': '<run-candidate-command-with-3x-budget>',
            'expected_artifact': 'runs/extended_budget.log',
        })

    elif ptype == 'objective_anchor':
        notes.append('Compare the current action against the original research contract before running more experiments.')
        commands.append({
            'name': 'objective_anchor_check',
            'command_template': 'scitriage check-objective <trace.json> <research_contract.md>',
            'expected_artifact': 'OBJECTIVE_ANCHOR_CHECK.md',
        })

    else:
        notes.append('No specialized command template is available for this probe type.')

    return {
        'trace_id': trace.trace_id,
        'status': 'probe_planned',
        'probe_type': ptype,
        'reason': probe.reason,
        'estimated_cost_gpu_hours': probe.estimated_cost_gpu_hours,
        'commands': commands,
        'notes': notes,
    }


def render_probe_plan_markdown(plan: Dict[str, object]) -> str:
    lines = [f"# Probe Plan: {plan['trace_id']}\n"]
    lines.append(f"- Status: `{plan['status']}`")
    if 'probe_type' in plan:
        lines.append(f"- Probe type: `{plan['probe_type']}`")
        lines.append(f"- Estimated cost: {plan.get('estimated_cost_gpu_hours')} GPU-hours")
        lines.append(f"- Reason: {plan.get('reason')}")
    lines.append('\n## Notes\n')
    for note in plan.get('notes', []):
        lines.append(f'- {note}')
    lines.append('\n## Command Templates\n')
    commands = plan.get('commands', [])
    if not commands:
        lines.append('- none')
    for command in commands:
        lines.append(f"### {command['name']}")
        lines.append('```bash')
        lines.append(command['command_template'])
        lines.append('```')
        lines.append(f"Expected artifact: `{command['expected_artifact']}`\n")
    return '\n'.join(lines)
