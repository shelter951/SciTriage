from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from textwrap import dedent
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scitriage.semantic_gate import select_with_semantic_gate


DEFAULT_VARIANTS = [
    "tiny_random_valid",
    "tiny_wider_valid",
    "invalid_missing_output",
    "invalid_config_only",
    "invalid_tokenizer_only",
]


def _candidate_source(variant: str) -> str:
    common = dedent(
        """
    from pathlib import Path
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    work = Path(".")
    train_files = list((work / "babylm_data" / "babylm_10M").glob("*.train"))[:2]
    tok_dir = work / "tok_train"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [str(p) for p in train_files],
        vocab_size=512,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    )
    tokenizer.save_model(str(tok_dir))
    fast = GPT2TokenizerFast.from_pretrained(
        str(tok_dir),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=64,
    )
    """
    ).strip() + "\n"
    if variant == "tiny_random_valid":
        return common + dedent(
            """
            config = GPT2Config(
                vocab_size=fast.vocab_size,
                n_positions=64,
                n_ctx=64,
                n_embd=32,
                n_layer=1,
                n_head=2,
                bos_token_id=fast.bos_token_id,
                eos_token_id=fast.eos_token_id,
                pad_token_id=fast.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
            out = work / "output"
            out.mkdir(exist_ok=True)
            model.save_pretrained(out)
            fast.save_pretrained(out)
            """
        ).strip() + "\n"
    if variant == "tiny_wider_valid":
        return common + dedent(
            """
            config = GPT2Config(
                vocab_size=fast.vocab_size,
                n_positions=64,
                n_ctx=64,
                n_embd=48,
                n_layer=1,
                n_head=3,
                bos_token_id=fast.bos_token_id,
                eos_token_id=fast.eos_token_id,
                pad_token_id=fast.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
            out = work / "output"
            out.mkdir(exist_ok=True)
            model.save_pretrained(out)
            fast.save_pretrained(out)
            """
        ).strip() + "\n"
    if variant == "invalid_missing_output":
        return "print('no output artifact written')\n"
    if variant == "invalid_config_only":
        return dedent(
            """
            import json
            from pathlib import Path

            out = Path("output")
            out.mkdir(exist_ok=True)
            (out / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
            """
        ).strip() + "\n"
    if variant == "invalid_tokenizer_only":
        return common + "fast.save_pretrained(work / 'output')\n"
    raise ValueError(f"unknown variant {variant}")


def _run_candidate(python: str, work_dir: Path, timeout: int) -> Dict[str, object]:
    output = work_dir / "output"
    if output.exists():
        shutil.rmtree(output)
    try:
        proc = subprocess.run(
            [python, "train.py"],
            cwd=work_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + f"\ntimeout_after_seconds: {timeout}",
            "output_exists": output.exists(),
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_exists": output.exists(),
    }


def _artifact_gate(work_dir: Path) -> Dict[str, object]:
    output = work_dir / "output"
    if not output.exists():
        return {"passed": False, "reason": "missing_output_dir"}
    try:
        AutoTokenizer.from_pretrained(output)
        AutoModelForCausalLM.from_pretrained(output)
    except Exception as exc:
        return {"passed": False, "reason": f"unloadable_output:{type(exc).__name__}:{exc}"}
    return {"passed": True, "reason": "loadable_model_and_tokenizer"}


def _read_eval_texts(work_dir: Path, max_eval_texts: int) -> List[str]:
    test_dir = work_dir / "babylm_data" / "babylm_test"
    texts: List[str] = []
    for path in sorted(test_dir.glob("*.test")):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                if len(texts) >= max_eval_texts:
                    return texts
    return texts


def _compatible_perplexity(work_dir: Path, max_eval_texts: int, batch_size: int) -> float | None:
    gate = _artifact_gate(work_dir)
    if not gate["passed"]:
        return None
    output = work_dir / "output"
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = _read_eval_texts(work_dir, max_eval_texts)
    if not texts:
        return None
    block_size = min(int(getattr(tokenizer, "model_max_length", 64) or 64), 64)
    encoded = tokenizer(texts, add_special_tokens=True)["input_ids"]
    token_stream = [tok for row in encoded for tok in row]
    usable = (len(token_stream) // block_size) * block_size
    if usable < block_size:
        return None
    chunks = [
        token_stream[i : i + block_size]
        for i in range(0, usable, block_size)
    ]
    losses = []
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = torch.tensor(chunks[i : i + batch_size], dtype=torch.long, device=device)
            loss = model(batch, labels=batch).loss
            losses.append(float(loss.detach().cpu()))
    return float(math.exp(mean(losses))) if losses else None


def _run_repeated(
    python: str,
    work_dir: Path,
    timeout: int,
    repeats: int,
    max_eval_texts: int,
    batch_size: int,
) -> Dict[str, object]:
    runs = []
    for _ in range(repeats):
        run = _run_candidate(python, work_dir, timeout)
        run["artifact_gate"] = _artifact_gate(work_dir)
        run["official_compatible_score"] = (
            _compatible_perplexity(work_dir, max_eval_texts, batch_size)
            if run["artifact_gate"]["passed"]
            else None
        )
        runs.append(run)
    scores = [r["official_compatible_score"] for r in runs if r["official_compatible_score"] is not None]
    return {
        "runs": runs,
        "official_score": mean(scores) if scores else None,
        "official_score_std": stdev(scores) if len(scores) >= 2 else 0.0,
        "official_score_min": min(scores) if scores else None,
        "official_score_max": max(scores) if scores else None,
        "returncode": 0 if len(scores) == repeats else 1,
        "stdout": "\n".join(str(r["stdout"]) for r in runs),
        "stderr": "\n".join(str(r["stderr"]) for r in runs),
        "artifact_gate_passed": all(bool(r["artifact_gate"]["passed"]) for r in runs),
        "artifact_gate_reasons": [r["artifact_gate"]["reason"] for r in runs],
    }


def _render_markdown(result: Dict[str, object]) -> str:
    rows: List[Dict[str, object]] = result["rows"]
    scored = [r for r in rows if r["official_score"] is not None]
    official_sorted = sorted(scored, key=lambda r: r["official_score"])
    valid = [r for r in scored if r["artifact_gate_passed"]]
    triage_sorted = sorted(valid, key=lambda r: r["official_score"])
    lines = ["# MLAgentBench BabyLM Candidate Audit\n"]
    lines.append("External task: `MLAgentBench/benchmarks/babylm`.")
    lines.append("Score is compatible causal-LM perplexity on BabyLM test text, lower is better.")
    lines.append(
        "The upstream MLAgentBench BabyLM `eval.py` is incompatible with the current "
        "server dependency stack (`transformers` 5.x and `datasets` 4.x remove APIs it imports). "
        "This audit therefore uses the same task data and perplexity objective with a compatibility evaluator, "
        "and marks that limitation explicitly.\n"
    )
    if official_sorted:
        lines.append(f"- Visible-score best candidate: `{official_sorted[0]['variant']}`")
    if triage_sorted:
        lines.append(f"- SciTriage-gated best candidate: `{triage_sorted[0]['variant']}`")
    blocked = [r for r in rows if not r["artifact_gate_passed"]]
    lines.append(f"- Candidates blocked by artifact gate: {len(blocked)}")
    lines.append(f"- Repeats per candidate: {result['repeats']}")
    lines.append(f"- Eval texts: {result['max_eval_texts']}")
    lines.append("\n## Policy Comparison\n")
    policy = result["policy_comparison"]
    lines.append("| Policy | Selected | Perplexity | Artifact Gate |")
    lines.append("|---|---|---:|---|")
    for name in ["visible_score_only", "scitriage_gated"]:
        item = policy[name]
        score = "-" if item["official_score"] is None else f"{item['official_score']:.4f}"
        lines.append(f"| `{name}` | `{item['selected']}` | {score} | {item['artifact_gate_passed']} |")
    lines.append("\n## Candidate Table\n")
    lines.append("| Variant | Mean Perplexity | Std | Artifact Gate | Triage Status |")
    lines.append("|---|---:|---:|---|---|")
    for row in sorted(rows, key=lambda r: (r["official_score"] is None, r["official_score"] or 1e99)):
        score = "-" if row["official_score"] is None else f"{row['official_score']:.4f}"
        std = "-" if row["official_score_std"] is None else f"{row['official_score_std']:.4f}"
        status = "allowed" if row["artifact_gate_passed"] else "blocked"
        lines.append(f"| `{row['variant']}` | {score} | {std} | {row['artifact_gate_passed']} | `{status}` |")
    lines.append("\n## Interpretation\n")
    lines.append(
        "This task adds a model-artifact validity surface: a language-model benchmark can only support "
        "a research claim if the candidate writes a loadable model and tokenizer. SciTriage blocks "
        "missing or partial artifacts before the agent can present an unloadable run as a result."
    )
    return "\n".join(lines)


def _link_or_copy_data(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagentbench-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-eval-texts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    root = Path(args.mlagentbench_root).expanduser().resolve()
    env_dir = root / "MLAgentBench" / "benchmarks" / "babylm" / "env"
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in args.variants:
        work = out / "workspaces" / variant
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        _link_or_copy_data(env_dir / "babylm_data", work / "babylm_data")
        shutil.copy2(env_dir / "babyLM_for_hf.py", work / "babyLM_for_hf.py")
        (work / "train.py").write_text(_candidate_source(variant))
        run = _run_repeated(
            args.python,
            work,
            args.timeout,
            args.repeats,
            args.max_eval_texts,
            args.batch_size,
        )
        row = {
            "variant": variant,
            **run,
        }
        rows.append(row)

    policy = select_with_semantic_gate(
        rows,
        score_key="official_score",
        invariant_key="artifact_gate_passed",
        lower_is_better=True,
    )
    result = {
        "task": "MLAgentBench/babylm",
        "metric": "compatible_perplexity",
        "lower_is_better": True,
        "repeats": args.repeats,
        "max_eval_texts": args.max_eval_texts,
        "rows": rows,
        "policy_comparison": {
            "visible_score_only": {
                "selected": policy["visible_score_only"]["selected"] if policy["visible_score_only"] else None,
                "official_score": policy["visible_score_only"]["score"] if policy["visible_score_only"] else None,
                "artifact_gate_passed": (
                    policy["visible_score_only"]["semantic_invariant_passed"]
                    if policy["visible_score_only"]
                    else False
                ),
            },
            "scitriage_gated": {
                "selected": policy["scitriage_gated"]["selected"] if policy["scitriage_gated"] else None,
                "official_score": policy["scitriage_gated"]["score"] if policy["scitriage_gated"] else None,
                "artifact_gate_passed": (
                    policy["scitriage_gated"]["semantic_invariant_passed"]
                    if policy["scitriage_gated"]
                    else False
                ),
            },
            "visible_winner_blocked": policy["visible_winner_blocked"],
            "blocked_candidates": policy["blocked_candidates"],
        },
        "compatibility_note": (
            "Upstream BabyLM eval.py imports removed transformers APIs and uses a dataset script "
            "loading path disabled by the installed datasets version. This script keeps the task data "
            "and perplexity objective but uses a local compatibility evaluator."
        ),
    }
    (out / "candidate_audit.json").write_text(json.dumps(result, indent=2))
    (out / "CANDIDATE_AUDIT.md").write_text(_render_markdown(result))
    print(out / "candidate_audit.json")
    print(out / "CANDIDATE_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
