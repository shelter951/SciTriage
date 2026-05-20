from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict


OOM_PATTERNS = [
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"torch\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
]

DEFAULT_AUTORESEARCH_PRESET = {
    "name": "rtx4090_quick_probe",
    "purpose": "produce cheap diagnostic evidence on RTX 4090 instead of imitating the H100 default run",
    "prepare.py": {
        "MAX_SEQ_LEN": 1024,
        "TIME_BUDGET": 60,
        "EVAL_TOKENS": "2 * 65536",
    },
    "train.py": {
        "DEPTH": 4,
        "DEVICE_BATCH_SIZE": 32,
        "TOTAL_BATCH_SIZE": "2**16",
        "WINDOW_PATTERN": "\"L\"",
    },
    "expected_effect": [
        "fits within 24GB VRAM on a single RTX 4090",
        "keeps val_bpb evaluation cheap enough for multiple seeds",
        "is a diagnostic probe, not a paper-quality final comparison",
    ],
}


def diagnose_resource_fit(run_log: str | Path) -> Dict[str, object]:
    path = Path(run_log).expanduser().resolve()
    text = path.read_text(errors="replace")
    oom = any(pattern.search(text) for pattern in OOM_PATTERNS)
    result: Dict[str, object] = {
        "run_log": str(path),
        "status": "oom_or_resource_mismatch" if oom else "no_resource_blocker_detected",
        "signals": [],
        "recommended_preset": DEFAULT_AUTORESEARCH_PRESET if oom else None,
    }
    if oom:
        result["signals"].append("training log contains CUDA OOM / torch.OutOfMemoryError")
        result["interpretation"] = (
            "The experiment reached the training forward pass but the default autoresearch "
            "configuration is too large for the observed GPU memory. Treat this as a harness "
            "resource-fit issue before judging any research idea."
        )
    else:
        result["interpretation"] = "No explicit OOM signature was found in the provided log."
    return result


def render_resource_fit(result: Dict[str, object]) -> str:
    lines = [f"# Resource Fit: {result['status']}\n"]
    lines.append(f"- Run log: `{result['run_log']}`")
    lines.append(f"- Interpretation: {result['interpretation']}")
    lines.append("\n## Signals\n")
    signals = result.get("signals", [])
    if not signals:
        lines.append("- none")
    for signal in signals:
        lines.append(f"- {signal}")
    preset = result.get("recommended_preset")
    if preset:
        lines.append("\n## Recommended Preset\n")
        lines.append("```json")
        lines.append(json.dumps(preset, indent=2))
        lines.append("```")
        lines.append(
            "\nThis preset is meant for triage probes and regression checks. "
            "Final claims still need the original evaluation contract or a clearly stated "
            "resource-normalized contract."
        )
    return "\n".join(lines)
