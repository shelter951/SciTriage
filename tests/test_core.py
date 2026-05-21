import json
import tempfile
import unittest
from pathlib import Path

from scitriage.aggregate import compare_seed_groups
from scitriage.aggregate import parse_metric
from scitriage.claim_gate import gate_claim
from scitriage.hooks import write_autoresearch_hook
from scitriage.plugin import assess_trace, seed_group_gate
from scitriage.evidence_board import build_evidence_board
from scitriage.adapters.filesystem import trace_from_filesystem
from scitriage.probe_priority import prioritize_probe
from scitriage.rules import diagnose
from scitriage.schema import MetricObservation, ResearchTrace


class CoreBehaviorTests(unittest.TestCase):
    def test_parse_metric_handles_benchmark_log_formats(self):
        self.assertEqual(parse_metric("score = -1.25e-03", "score"), -1.25e-03)
        self.assertAlmostEqual(parse_metric('"accuracy": "87.5%"', "accuracy"), 0.875)
        self.assertEqual(parse_metric('{"submission_score": 12.5}', "submission_score"), 12.5)

    def test_noisy_one_shot_blocks_claim(self):
        trace = ResearchTrace(
            trace_id="case",
            question="Improve validation metric.",
            proposal="Tune one training hyperparameter.",
            claims=["The candidate significantly improves validation loss."],
            changed_files=["train.py"],
            logs="val_bpb: 0.99",
            metrics=[
                MetricObservation(
                    name="val_bpb",
                    candidate=0.99,
                    baseline=1.0,
                    higher_is_better=False,
                    baseline_std=0.02,
                    seeds=1,
                )
            ],
            experiment={"budget_minutes": 5},
        )
        report = diagnose(trace)
        self.assertIn("NOISY_RESULT", report.risk_labels)
        self.assertIn("UNSUPPORTED_CLAIM", report.risk_labels)
        self.assertEqual(report.recommended_probe.probe_type, "multi_seed_rerun")

    def test_group_compare_marks_noise_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = []
            candidate = []
            for idx, value in enumerate([1.00, 1.01, 0.99], 1):
                path = root / f"b{idx}.log"
                path.write_text(f"val_bpb: {value}\n")
                baseline.append(path)
            for idx, value in enumerate([0.995, 0.992, 1.000], 1):
                path = root / f"c{idx}.log"
                path.write_text(f"val_bpb: {value}\n")
                candidate.append(path)
            result = compare_seed_groups(baseline, candidate, "val_bpb")
            self.assertEqual(result["verdict"], "inconclusive_within_noise")

    def test_claim_gate_blocks_inconclusive_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare.json"
            path.write_text(json.dumps({
                "verdict": "inconclusive_within_noise",
                "delta_improvement": 0.002,
                "z_margin": 0.005,
            }))
            result = gate_claim(path, "The candidate improves val_bpb.")
            self.assertEqual(result["status"], "blocked")

    def test_probe_priority_uses_group_candidate_as_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "candidate": {"mean": 1.10, "std": 0.01}
            }))
            log = root / "candidate.log"
            log.write_text("val_bpb: 1.101\n")
            result = prioritize_probe(log, baseline, "val_bpb")
            self.assertEqual(result["priority"], "low")

    def test_plugin_assess_trace_returns_bundle(self):
        bundle = assess_trace({
            "trace_id": "plugin_case",
            "question": "Improve validation metric.",
            "proposal": "Tune training.",
            "claims": ["The candidate improves validation loss."],
            "changed_files": ["train.py"],
            "logs": "val_loss: 0.9",
            "metrics": [{
                "name": "val_loss",
                "candidate": 0.9,
                "baseline": 1.0,
                "higher_is_better": False,
                "seeds": 1,
            }],
            "experiment": {"budget_minutes": 5},
        })
        self.assertIn("report", bundle)
        self.assertIn("probe_plan", bundle)
        self.assertEqual(bundle["report"]["status"], "needs_probe")

    def test_plugin_seed_group_gate_allows_clear_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = []
            candidate = []
            for idx, value in enumerate([1.20, 1.21, 1.19], 1):
                path = root / f"b{idx}.log"
                path.write_text(f"val_bpb: {value}\n")
                baseline.append(path)
            for idx, value in enumerate([1.10, 1.11, 1.09], 1):
                path = root / f"c{idx}.log"
                path.write_text(f"val_bpb: {value}\n")
                candidate.append(path)
            bundle = seed_group_gate(baseline, candidate, "val_bpb", "The candidate improves val_bpb.")
            self.assertEqual(bundle["comparison"]["verdict"], "supports_improvement")
            self.assertEqual(bundle["claim_gate"]["status"], "allowed")

    def test_filesystem_adapter_reads_extensionless_benchmark_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "log").write_text("score: 0.42\n")
            agent_log = root / "agent_log"
            agent_log.mkdir()
            (agent_log / "main_log").write_text("Final Answer: improved model\n")
            env_log = root / "env_log"
            env_log.mkdir()
            (env_log / "trace.json").write_text('{"steps": []}')
            trace = trace_from_filesystem(root, trace_id="bench")
            self.assertGreaterEqual(trace.experiment["num_files_scanned"], 2)
            self.assertEqual(trace.metrics[0].name, "score")
            self.assertEqual(trace.metrics[0].candidate, 0.42)

    def test_hook_writer_creates_executable_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scitriage_after_run.py"
            result = write_autoresearch_hook(path)
            self.assertTrue(path.exists())
            self.assertIn("post-run SciTriage hook", result["purpose"])
            self.assertIn("assess_autoresearch_run", path.read_text())
            self.assertIn("--scitriage-root", path.read_text())

    def test_evidence_board_policy_saves_seed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = root / "sweep.json"
            sweep.write_text(json.dumps({
                "rows": [
                    {"variant": "big_win", "val_bpb": 0.8, "delta_vs_seed1_baseline": 0.2},
                    {"variant": "small_win", "val_bpb": 0.98, "delta_vs_seed1_baseline": 0.02},
                    {"variant": "loss", "val_bpb": 1.1, "delta_vs_seed1_baseline": -0.1},
                ]
            }))
            group_paths = []
            for variant, verdict in [
                ("big_win", "supports_improvement"),
                ("small_win", "inconclusive_within_noise"),
                ("loss", "does_not_support_improvement"),
            ]:
                path = root / f"{variant}_vs_baseline_group_compare.json"
                path.write_text(json.dumps({
                    "verdict": verdict,
                    "delta_improvement": 0.2 if variant == "big_win" else 0.01,
                    "z_margin": 0.05,
                    "baseline": {"std": 0.05},
                }))
                group_paths.append(path)
            board = build_evidence_board([sweep], group_paths)
            self.assertEqual(board["policy_eval"]["scitriage_high_priority_only"]["true_accepts"], 1)
            self.assertEqual(board["policy_eval"]["scitriage_high_priority_only"]["false_accepts"], 0)
            self.assertGreater(board["policy_eval"]["cost_savings_vs_verify_all_positives"], 0)
            self.assertEqual(board["policy_eval"]["scitriage_family_landmarks"]["true_family_accepts"], 1)
            self.assertEqual(board["policy_eval"]["scitriage_family_landmarks"]["family_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

