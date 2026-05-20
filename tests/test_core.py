import json
import tempfile
import unittest
from pathlib import Path

from scitriage.aggregate import compare_seed_groups
from scitriage.claim_gate import gate_claim
from scitriage.probe_priority import prioritize_probe
from scitriage.rules import diagnose
from scitriage.schema import MetricObservation, ResearchTrace


class CoreBehaviorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

