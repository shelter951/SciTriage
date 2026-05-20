from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict

from .resource_fit import DEFAULT_AUTORESEARCH_PRESET


def _replace_assignment(text: str, name: str, value: object) -> str:
    pattern = re.compile(rf"^{name}\s*=.*$", re.MULTILINE)
    replacement = f"{name} = {value}"
    if not pattern.search(text):
        raise ValueError(f"Could not find assignment for {name}")
    return pattern.sub(replacement, text, count=1)


def _patch_seed_control(text: str) -> str:
    old = "torch.manual_seed(42)\ntorch.cuda.manual_seed(42)"
    new = (
        "seed = int(os.environ.get(\"SCITRIAGE_SEED\", \"42\"))\n"
        "torch.manual_seed(seed)\n"
        "torch.cuda.manual_seed(seed)\n"
        "print(f\"scitriage_seed: {seed}\")"
    )
    if old not in text:
        return text
    return text.replace(old, new, 1)


def materialize_autoresearch_probe(source_repo: str | Path, target_dir: str | Path) -> Dict[str, object]:
    source = Path(source_repo).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    if not (source / "train.py").exists() or not (source / "prepare.py").exists():
        raise FileNotFoundError("source_repo must contain train.py and prepare.py")
    if target.exists():
        raise FileExistsError(f"target_dir already exists: {target}")

    ignore = shutil.ignore_patterns(".git", ".venv", "runs", "__pycache__", "*.pyc")
    shutil.copytree(source, target, ignore=ignore)

    prepare_path = target / "prepare.py"
    train_path = target / "train.py"
    prepare = prepare_path.read_text()
    train = train_path.read_text()

    prepare = _replace_assignment(prepare, "MAX_SEQ_LEN", DEFAULT_AUTORESEARCH_PRESET["prepare.py"]["MAX_SEQ_LEN"])
    prepare = _replace_assignment(prepare, "TIME_BUDGET", DEFAULT_AUTORESEARCH_PRESET["prepare.py"]["TIME_BUDGET"])
    prepare = _replace_assignment(prepare, "EVAL_TOKENS", DEFAULT_AUTORESEARCH_PRESET["prepare.py"]["EVAL_TOKENS"])

    train = _replace_assignment(train, "WINDOW_PATTERN", DEFAULT_AUTORESEARCH_PRESET["train.py"]["WINDOW_PATTERN"])
    train = _replace_assignment(train, "TOTAL_BATCH_SIZE", DEFAULT_AUTORESEARCH_PRESET["train.py"]["TOTAL_BATCH_SIZE"])
    train = _replace_assignment(train, "DEPTH", DEFAULT_AUTORESEARCH_PRESET["train.py"]["DEPTH"])
    train = _replace_assignment(train, "DEVICE_BATCH_SIZE", DEFAULT_AUTORESEARCH_PRESET["train.py"]["DEVICE_BATCH_SIZE"])
    train = _patch_seed_control(train)

    prepare_path.write_text(prepare)
    train_path.write_text(train)

    manifest = {
        "source_repo": str(source),
        "target_dir": str(target),
        "preset": DEFAULT_AUTORESEARCH_PRESET,
        "commands": [
            "export HF_ENDPOINT=https://hf-mirror.com",
            "CUDA_VISIBLE_DEVICES=4 SCITRIAGE_SEED=1 uv run train.py > runs/seed_1.log 2>&1",
            "CUDA_VISIBLE_DEVICES=4 SCITRIAGE_SEED=2 uv run train.py > runs/seed_2.log 2>&1",
            "CUDA_VISIBLE_DEVICES=4 SCITRIAGE_SEED=3 uv run train.py > runs/seed_3.log 2>&1",
        ],
        "notes": [
            "This is a diagnostic probe copy; it intentionally excludes the original .venv and runs directories.",
            "The preset changes the resource contract, so its results should be reported as probe evidence.",
        ],
    }
    (target / "SCITRIAGE_PROBE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (target / "runs").mkdir(exist_ok=True)
    return manifest


def render_probe_manifest(manifest: Dict[str, object]) -> str:
    lines = [f"# Autoresearch Probe: {manifest['target_dir']}\n"]
    lines.append(f"- Source repo: `{manifest['source_repo']}`")
    lines.append("- Preset: `rtx4090_quick_probe`")
    lines.append("\n## Commands\n")
    for command in manifest["commands"]:
        lines.append("```bash")
        lines.append(command)
        lines.append("```")
    lines.append("\n## Notes\n")
    for note in manifest["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)
