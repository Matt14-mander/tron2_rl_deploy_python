"""Unit checks for safe stop/restart without the native SDK or MuJoCo."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "controllers" / "sfyg_state_machine.py"
SPEC = importlib.util.spec_from_file_location("sfyg_gait_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SafeGaitTest(unittest.TestCase):
    def setUp(self):
        self.floor = np.array([-0.29, -0.08, 0.0])
        self.gait = MODULE.SafeGaitStateMachine(self.floor)

    @staticmethod
    def support(now, speed=0.02, left=100, right=100):
        return MODULE.SupportState(now, speed, left, right)

    def test_zero_command_is_replaced_by_nonzero_floor(self):
        command, mode, weight = self.gait.update(
            np.zeros(3), None, 0.02, np.zeros(3), 0.0)
        self.assertEqual(mode, "BRAKE")
        np.testing.assert_allclose(command, self.floor)
        self.assertEqual(weight, 1.0)

    def test_rate_limit_and_timeout_never_emit_zero(self):
        self.gait.update([-0.5, 0, 0], None, 0.02, np.zeros(3), 0.0)
        before, _, _ = self.gait.update([-0.5, 0, 0], None, 0.02, np.zeros(3), 0.02)
        self.gait.solver_timeout()
        after, mode, _ = self.gait.update(np.zeros(3), None, 0.02, np.zeros(3), 0.04)
        self.assertEqual(mode, "BRAKE")
        self.assertLessEqual(np.max(np.abs(after - before)), 0.00501)
        self.assertGreater(np.linalg.norm(after[:2]), 0.02)

    def test_pd_stand_requires_fresh_double_support_and_dwell(self):
        gait = MODULE.SafeGaitStateMachine(self.floor, allow_pd_stand=True)
        for i in range(50):
            now = i * 0.02
            _, mode, _ = gait.update(np.zeros(3), self.support(now, right=0),
                                     0.02, np.zeros(3), now)
            self.assertEqual(mode, "BRAKE")
        for i in range(50, 81):
            now = i * 0.02
            _, mode, _ = gait.update(np.zeros(3), self.support(now),
                                     0.02, np.zeros(3), now)
        self.assertEqual(mode, "STAND_ENTRY")
        _, mode, weight = gait.update(np.zeros(3), None, 0.02, np.zeros(3), 1.64)
        self.assertEqual(mode, "WALK_ENTRY")
        self.assertLessEqual(weight, 1.0)

    def test_stand_to_walk_has_smooth_policy_weight(self):
        gait = MODULE.SafeGaitStateMachine(self.floor, allow_pd_stand=True)
        for i in range(80):
            now = i * 0.02
            gait.update(np.zeros(3), self.support(now), 0.02, np.zeros(3), now)
        for i in range(80, 122):
            now = i * 0.02
            gait.update(np.zeros(3), self.support(now), 0.02, np.zeros(3), now)
        self.assertEqual(gait.mode, "STAND")
        _, mode, weight = gait.update(np.zeros(3), self.support(2.44),
                                      0.02, np.zeros(3), 2.44, resume=True)
        self.assertEqual(mode, "WALK_ENTRY")
        self.assertEqual(weight, 0.0)
        _, _, weight = gait.update(self.floor, self.support(2.84),
                                   0.02, np.zeros(3), 2.84)
        self.assertAlmostEqual(weight, 0.5)


if __name__ == "__main__":
    unittest.main()
