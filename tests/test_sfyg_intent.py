import importlib.util
import json
import tempfile
import unittest
import threading
import io
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "intent_adapter", Path(__file__).resolve().parents[1] / "controllers/sfyg_intent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntentTest(unittest.TestCase):
    def test_clock_keeps_phase_and_skips_missing_frames(self):
        clock = module.ObservationClock(0.)
        self.assertTrue(clock.due(0.))
        self.assertFalse(clock.due(.019))
        self.assertTrue(clock.due(.023))
        self.assertAlmostEqual(clock.next_time, .04)
        self.assertTrue(clock.due(.091))
        self.assertAlmostEqual(clock.next_time, .1)
        self.assertFalse(clock.due(.092))

    def test_async_worker_copies_telemetry_and_reports_overflow(self):
        entered, release = threading.Event(), threading.Event()
        class FakeObserver:
            def __init__(self, *_):
                self.items = []
                self.log = io.StringIO()
                self.closed = False
            def update(self, *item):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test worker timeout")
                self.items.append(item)
            def close(self):
                self.closed = True
        with patch.object(module, "IntentObserver", FakeObserver):
            observer = module.AsyncIntentObserver("unused", "unused")
            q = np.zeros(18)
            observer.update(0, 1, 0, q, q, np.zeros(6), None)
            self.assertTrue(entered.wait(2))
            q.fill(5)
            for t in [.02, .04, .06]:
                observer.update(t, 2, t, q, q, np.zeros(6), None)
            self.assertEqual(observer.dropped, 1)
            release.set()
            observer.close()
            self.assertFalse(observer.thread.is_alive())
            self.assertTrue(observer.observer.closed)
            self.assertEqual(len(observer.observer.items), 3)
            self.assertTrue(np.all(observer.observer.items[0][3] == 0))
            self.assertIn('"queue_dropped_samples": 1', observer.observer.log.getvalue())

    def test_history_previous_target_and_discontinuities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = {"schema": "paint_pd_response_v1", "history": 4,
                "input_size": 72, "step_dt": .02,
                "joint_names": [f"arm{i}_Joint" for i in range(1, 7)],
                "output_frame": "payload_local", "output_point": "payload_COM",
                "output_order": ["Fx_N", "Fy_N", "Mz_Nm"],
                "input_units": "rad, rad/s, rad",
                "output_semantics": "external_on_payload during preceding control interval"}
            (root / "metadata.json").write_text(json.dumps(meta))
            net = torch.nn.Linear(72, 3)
            with torch.no_grad():
                net.weight.fill_(1)
                net.bias.zero_()
            torch.jit.save(torch.jit.script(net), str(root / "intent_estimator.ts"))
            observer = module.IntentObserver(root, root / "log.jsonl")
            solution = SimpleNamespace(wrench_prediction=np.ones((5, 6)),
                arm_velocity=np.zeros(6), arm_effort=np.zeros(6), base_command=np.zeros(3))
            for i in range(4):
                observer.update(10 + i * .02, i, 10 + i * .02,
                                np.zeros(18), np.zeros(18), np.ones(6), solution)
            observer.update(10.08, 3, 10.08, np.zeros(18), np.zeros(18), np.ones(6), solution)
            observer.update(10.2, 4, 10.2, np.zeros(18), np.zeros(18), np.ones(6), solution)
            observer.close()
            rows = [json.loads(s) for s in (root / "log.jsonl").read_text().splitlines()][1:]
            self.assertEqual([r["valid"] for r in rows], [False, False, False, True, False, False])
            self.assertEqual(rows[3]["interaction_wrench_payload_local"], [24., 24., 24.])
            self.assertEqual(rows[4]["history_reset_reason"], "stale_or_repeated_state")
            self.assertEqual(rows[5]["history_reset_reason"], "sample_time_gap")
            self.assertTrue(np.all(solution.wrench_prediction == 1))


if __name__ == "__main__":
    unittest.main()
