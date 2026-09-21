"""Pure contract tests for SFYG deployment; no SDK or ONNX runtime required."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "controllers" / "sfyg_contract.py"
SPEC = importlib.util.spec_from_file_location("sfyg_contract_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SfygContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = MODULE.SFYGPolicyConfig(
            REPO / "controllers" / "model" / "SFYG_TRON2A" / "params.yaml"
        )

    def test_dimensions_and_wire_order(self):
        self.assertEqual(self.config.history_size, 420)
        self.assertEqual(self.config.policy_input_size, 78)
        self.assertEqual(self.config.wire_joint_names[10], "arm1_Joint")
        self.assertEqual(self.config.wire_joint_names[-1], "gripper2_Joint")

    def test_policy_input_contains_scaled_wrench_in_contract_order(self):
        proprio = MODULE.build_proprio_observation(
            self.config,
            self.config.default_q,
            np.zeros(18),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(10),
            np.asarray((0.2, 0.0, 0.0)),
            0.25,
        )
        wrench = np.ones((5, 6))
        policy_input = MODULE.compose_policy_input(
            self.config, np.zeros(3), proprio, wrench, np.zeros(3)
        )
        self.assertEqual(policy_input.shape, (78,))
        np.testing.assert_allclose(
            policy_input[45:75].reshape(5, 6)[0], MODULE.WRENCH_SCALE
        )

    def test_ocs2_arm_targets_do_not_replace_leg_policy_targets(self):
        solution = MODULE.Ocs2Solution(
            0.0,
            np.arange(6, dtype=float),
            np.zeros(6),
            np.zeros(6),
            np.zeros(3),
            np.zeros((5, 6)),
        )
        actions = np.ones(10)
        targets = MODULE.compose_joint_targets(self.config, actions, solution)
        np.testing.assert_allclose(
            targets[:10], self.config.default_q[:10] + self.config.action_scale
        )
        np.testing.assert_allclose(targets[10:16], solution.arm_position)
        np.testing.assert_allclose(targets[16:], self.config.default_q[16:])

    def test_trajectory_terminal_hold_zeros_arm_and_base_velocity(self):
        rows = np.zeros((2, 52))
        rows[:, 0] = (0.0, 1.0)
        rows[:, 1:7] = 0.5
        rows[:, 7:13] = 2.0
        rows[:, 19:22] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            np.savetxt(
                path,
                rows,
                delimiter=",",
                header=",".join(MODULE.TRAJECTORY_COLUMNS),
                comments="",
            )
            solution = MODULE.Ocs2Trajectory(path).sample(2.0)
        np.testing.assert_allclose(solution.arm_velocity, 0.0)
        np.testing.assert_allclose(solution.base_command, 0.0)


if __name__ == "__main__":
    unittest.main()
