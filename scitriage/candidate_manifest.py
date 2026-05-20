from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def build_candidate_manifest(
    variant_id: str,
    family: str,
    workspace: str,
    base_variant: str,
    patch_intent: str,
    claim: str,
    changed_files: Iterable[str],
    seed_logs: Iterable[str],
    notes: str = "",
) -> Dict[str, object]:
    return {
        "variant_id": variant_id,
        "family": family,
        "workspace": workspace,
        "base_variant": base_variant,
        "patch_intent": patch_intent,
        "claim": claim,
        "changed_files": list(changed_files),
        "seed_logs": list(seed_logs),
        "notes": notes,
    }


def write_candidate_manifest(manifest: Dict[str, object], out: str | Path) -> Dict[str, object]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def summarize_candidate_manifests(paths: Iterable[str | Path]) -> Dict[str, object]:
    manifests: List[Dict[str, object]] = []
    for raw in paths:
        manifests.append(json.loads(Path(raw).read_text()))
    by_family: Dict[str, int] = {}
    for item in manifests:
        family = item.get("family", "unknown")
        by_family[family] = by_family.get(family, 0) + 1
    return {
        "num_candidates": len(manifests),
        "families": by_family,
        "candidates": manifests,
    }


def render_candidate_summary_markdown(summary: Dict[str, object]) -> str:
    lines = ["# Candidate Manifest Summary\n"]
    lines.append(f"- Candidates: {summary['num_candidates']}")
    lines.append("\n## Families\n")
    for family, count in sorted(summary["families"].items()):
        lines.append(f"- `{family}`: {count}")
    lines.append("\n## Candidates\n")
    lines.append("| Variant | Family | Base | Patch Intent | Claim |")
    lines.append("|---|---|---|---|---|")
    for item in summary["candidates"]:
        lines.append(
            f"| {item['variant_id']} | {item['family']} | {item['base_variant']} | "
            f"{item['patch_intent']} | {item['claim']} |"
        )
    return "\n".join(lines)
