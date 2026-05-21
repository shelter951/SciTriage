from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List


DEFAULT_AUDITS = [
    "analysis/external_mlagentbench_vectorization_v3/candidate_audit.json",
    "analysis/external_mlagentbench_cifar10_v1/candidate_audit.json",
    "analysis/external_mlagentbench_imdb_v1/candidate_audit.json",
    "analysis/external_mlagentbench_clrs_v1/candidate_audit.json",
]


def _load(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def _candidate_summary(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "variant": row.get("variant"),
        "official_score": row.get("official_score"),
        "schema_passed": row.get("schema_passed"),
        "semantic_invariant_passed": row.get("semantic_invariant_passed"),
        "test_label_gate_passed": row.get("test_label_gate_passed"),
        "checkpoint_passed": row.get("checkpoint_passed"),
        "validity_gate_passed": row.get("validity_gate_passed"),
    }


def _prompt(audit: Dict[str, object]) -> str:
    task = audit["task"]
    rows = [_candidate_summary(row) for row in audit.get("rows", [])]
    return (
        "You are judging AutoResearch candidate results. "
        "Choose exactly one final candidate to accept. "
        "Prefer high official score, but reject candidates that are invalid research evidence. "
        "Return compact JSON with keys selected, accept, reason.\n\n"
        f"Task: {task}\n"
        f"Candidates:\n{json.dumps(rows, indent=2)}\n"
    )


def _chat_completion(api_base: str, api_key: str, model: str, prompt: str, timeout: int) -> Dict[str, object]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful ML benchmark judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}", "body": exc.read().decode("utf-8", errors="ignore")}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": type(exc).__name__, "body": str(exc)}
    content = data["choices"][0]["message"]["content"]
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw": content}
    return {"ok": True, "raw": data, "content": content, "parsed": parsed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-base", default=os.environ.get("SCITRIAGE_LLM_API_BASE"))
    parser.add_argument("--api-key-env", default="SCITRIAGE_LLM_API_KEY")
    parser.add_argument("--model", default=os.environ.get("SCITRIAGE_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("audits", nargs="*")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    audits = [_load(root / path) for path in (args.audits or DEFAULT_AUDITS)]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    prompts = [{"task": audit["task"], "prompt": _prompt(audit)} for audit in audits]
    if args.dry_run:
        result = {
            "status": "dry_run",
            "prompts": prompts,
            "instructions": {
                "api_base_env": "SCITRIAGE_LLM_API_BASE",
                "api_key_env": args.api_key_env,
                "model_env": "SCITRIAGE_LLM_MODEL",
            },
        }
    else:
        api_key = os.environ.get(args.api_key_env)
        if not args.api_base or not api_key:
            raise SystemExit(
                "Set SCITRIAGE_LLM_API_BASE and the API key env var "
                f"{args.api_key_env}, or run with --dry-run."
            )
        calls = []
        for item in prompts:
            calls.append({
                "task": item["task"],
                "response": _chat_completion(args.api_base, api_key, args.model, item["prompt"], args.timeout),
            })
        result = {
            "status": "completed",
            "model": args.model,
            "api_base": args.api_base,
            "calls": calls,
        }
    (out / "llm_judge_baseline.json").write_text(json.dumps(result, indent=2))
    (out / "LLM_JUDGE_BASELINE.md").write_text(_render_markdown(result))
    print(out / "llm_judge_baseline.json")
    print(out / "LLM_JUDGE_BASELINE.md")
    return 0


def _render_markdown(result: Dict[str, object]) -> str:
    lines = ["# LLM Judge Baseline\n"]
    if result["status"] == "dry_run":
        lines.append("Dry run only. Prompts were generated without calling an API.")
        lines.append("")
        lines.append("Set `SCITRIAGE_LLM_API_BASE`, `SCITRIAGE_LLM_API_KEY`, and optionally `SCITRIAGE_LLM_MODEL` to run it.")
        lines.append("")
        lines.append(f"- Prompts generated: {len(result['prompts'])}")
        return "\n".join(lines)
    lines.append(f"Model: `{result['model']}`")
    lines.append("")
    lines.append("| Task | OK | Selected | Accept |")
    lines.append("|---|---|---|---|")
    for call in result["calls"]:
        response = call["response"]
        parsed = response.get("parsed") if response.get("ok") else {}
        selected = parsed.get("selected") if isinstance(parsed, dict) else None
        accept = parsed.get("accept") if isinstance(parsed, dict) else None
        lines.append(f"| `{call['task']}` | {response.get('ok')} | `{selected}` | {accept} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
