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

    def test_zero_velocity_command_keeps_training_time_phase(self):
        elapsed = 0.25
        proprio = MODULE.build_proprio_observation(
            self.config,
            self.config.default_q,
            np.zeros(18),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(10),
            np.zeros(3),
            elapsed,
        )
        angle = 2.0 * np.pi * ((elapsed * self.config.gait[0]) % 1.0)
        np.testing.assert_allclose(
            proprio[36:38], (np.sin(angle), np.cos(angle)), atol=1.0e-6
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

    def test_joint_space_arm_test_moves_one_joint_and_returns_smoothly(self):
        origin = np.array([0.0, np.pi / 2, -1.4835299, 0.0, 0.0, 0.0])
        motion = MODULE.JointSpaceArmTest(origin, "arm2", 0.1)
        base_command = np.array([-0.29, -0.08, 0.0])
        warmup = motion.sample(2.0, base_command)
        outbound = motion.sample(4.0, base_command)
        hold = motion.sample(5.5, base_command)
        returning = motion.sample(7.0, base_command)
        complete = motion.sample(9.0, base_command)

        np.testing.assert_allclose(warmup.arm_position, origin)
        np.testing.assert_allclose(warmup.arm_velocity, 0.0)
        self.assertAlmostEqual(outbound.arm_position[1], origin[1] + 0.05)
        self.assertAlmostEqual(outbound.arm_velocity[1], 0.09375)
        np.testing.assert_allclose(hold.arm_position[1], origin[1] + 0.1)
        np.testing.assert_allclose(hold.arm_velocity, 0.0)
        self.assertAlmostEqual(returning.arm_position[1], origin[1] + 0.05)
        self.assertAlmostEqual(returning.arm_velocity[1], -0.09375)
        np.testing.assert_allclose(complete.arm_position, origin)
        np.testing.assert_allclose(complete.arm_velocity, 0.0)
        for solution in (warmup, outbound, hold, returning, complete):
            np.testing.assert_allclose(solution.arm_position[[0, 2, 3, 4, 5]], origin[[0, 2, 3, 4, 5]])
            np.testing.assert_allclose(solution.base_command, base_command)
            np.testing.assert_allclose(solution.arm_effort, 0.0)
            np.testing.assert_allclose(solution.wrench_prediction, 0.0)

    def test_joint_space_arm_test_rejects_unsafe_requests(self):
        origin = np.array([0.0, np.pi / 2, -1.4835299, 0.0, 0.0, 0.0])
        for joint, delta, kwargs in (
            ("arm4", -2.385, {}),
            ("arm4", -0.2, {}),
            ("arm4", 0.0, {}),
            ("arm4", float("nan"), {}),
            ("arm4", 0.1, {"start_delay": 0.0}),
            ("arm4", 0.1, {"move_duration": 0.5}),
            ("arm4", 0.1, {"hold_duration": 0.0}),
        ):
            with self.subTest(joint=joint, delta=delta, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MODULE.JointSpaceArmTest(origin, joint, delta, **kwargs)
        near_limit = origin.copy()
        near_limit[3] = -1.5
        with self.assertRaisesRegex(ValueError, "limits"):
            MODULE.JointSpaceArmTest(near_limit, "arm4", -0.1)

    def test_two_joint_arm_test_moves_smoothly_and_returns(self):
        origin = np.array([0.0, np.pi / 2, -1.4835299, 0.0, 0.0, 0.0])
        offsets = np.array([0.0, 0.05, 0.0, -0.05, 0.0, 0.0])
        motion = MODULE.JointSpaceArmTest(origin, offsets=offsets)
        np.testing.assert_array_equal(motion.active_indices, [1, 3])
        command = np.array([-0.29, -0.08, 0.0])
        for elapsed, fraction, direction in (
            (2.0, 0.0, 0.0),
            (4.0, 0.5, 1.875 / 2.0),
            (5.5, 1.0, 0.0),
            (7.0, 0.5, -1.875 / 2.0),
            (9.0, 0.0, 0.0),
        ):
            with self.subTest(elapsed=elapsed):
                sample = motion.sample(elapsed, command)
                np.testing.assert_allclose(
                    sample.arm_position, origin + fraction * offsets
                )
                np.testing.assert_allclose(
                    sample.arm_velocity, direction * offsets
                )
                np.testing.assert_allclose(sample.arm_effort, 0.0)
                np.testing.assert_allclose(sample.wrench_prediction, 0.0)
                np.testing.assert_allclose(sample.base_command, command)

    def test_two_joint_arm_test_rejects_invalid_offsets(self):
        origin = np.array([0.0, np.pi / 2, -1.4835299, 0.0, 0.0, 0.0])
        for offsets in (
            [0, 0, 0, 0, 0, 0],
            [0, 0.05, 0, 0, 0, 0],
            [0, 0.05, 0, -0.05, 0.01, 0],
            [0, 0.11, 0, -0.01, 0, 0],
            [0, 0.10, 0, -0.10, 0, 0],
            [0, float("nan"), 0, -0.05, 0, 0],
            [0, 0.05, 0, -0.05, 0],
        ):
            with self.subTest(offsets=offsets):
                with self.assertRaises(ValueError):
                    MODULE.JointSpaceArmTest(origin, offsets=offsets)
        with self.assertRaises(ValueError):
            MODULE.JointSpaceArmTest(origin, "arm2", 0.05, offsets=[0, 0.05, 0, -0.05, 0, 0])
        near_limit = origin.copy()
        near_limit[3] = -1.5
        with self.assertRaisesRegex(ValueError, "limits"):
            MODULE.JointSpaceArmTest(
                near_limit, offsets=[0, 0.05, 0, -0.05, 0, 0]
            )

    def test_arm_test_guard_uses_static_sag_as_baseline(self):
        baseline = np.array([-0.03, 0.13, 0.09, 0.02, 0.002, 0.0])
        velocity = np.full(6, 0.07)
        guard = MODULE.arm_test_tracking_ok
        self.assertTrue(guard(baseline, velocity, 0.04))
        self.assertTrue(guard(baseline + 0.03, velocity, 0.04, baseline))
        self.assertFalse(guard(np.full(6, 0.46), velocity, 0.04))
        self.assertFalse(guard(baseline, np.full(6, 18.0), 0.04))
        self.assertFalse(guard(baseline + 0.13, velocity, 0.04, baseline))
        self.assertFalse(guard(baseline, velocity, 0.5, baseline))

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

    def test_repeated_terminal_arm_position_does_not_replay_nonzero_velocity(self):
        rows = np.zeros((3, 52))
        rows[:, 0] = (0.0, 1.0, 1.5)
        rows[0, 1:7] = 0.5
        rows[1:, 1:7] = 0.75
        rows[:2, 7:13] = 2.0
        rows[1, 19:22] = -0.2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            np.savetxt(
                path, rows, delimiter=",",
                header=",".join(MODULE.TRAJECTORY_COLUMNS), comments="",
            )
            trajectory = MODULE.Ocs2Trajectory(path)
            before_hold = trajectory.sample(0.5)
            hold_start = trajectory.sample(1.0)
            mid_hold = trajectory.sample(1.25)

        np.testing.assert_allclose(before_hold.arm_velocity, 2.0)
        np.testing.assert_allclose(hold_start.arm_velocity, 0.0)
        np.testing.assert_allclose(mid_hold.arm_position, 0.75)
        np.testing.assert_allclose(mid_hold.arm_velocity, 0.0)
        np.testing.assert_allclose(mid_hold.base_command, -0.1)

    def test_ocs2_replay_warmup_terminal_command_and_ablation(self):
        rows = np.zeros((2, 52))
        rows[:, 0] = (0.0, 1.0)
        rows[0, 1:7] = 0.5
        rows[1, 1:7] = 0.75
        rows[:, 7:13] = 2.0
        rows[:, 13:19] = 3.0
        rows[0, 19:22] = (-0.3, -0.08, 0.0)
        rows[1, 19:22] = (-0.6, 0.0, 0.0)
        rows[:, 22:52] = 4.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            np.savetxt(
                path, rows, delimiter=",",
                header=",".join(MODULE.TRAJECTORY_COLUMNS), comments="",
            )
            trajectory = MODULE.Ocs2Trajectory(path)
            arm_position = np.full(6, 0.25)
            options = dict(
                start_delay=1.0,
                terminal_command=np.array((-0.25, 0.0, 0.0)),
            )

            warmup = trajectory.sample_for_deployment(0.5, arm_position, **options)
            np.testing.assert_allclose(warmup.arm_position, arm_position)
            np.testing.assert_allclose(warmup.arm_velocity, 0.0)
            np.testing.assert_allclose(warmup.arm_effort, 0.0)
            np.testing.assert_allclose(warmup.base_command, rows[0, 19:22])
            np.testing.assert_allclose(warmup.wrench_prediction, 0.0)

            motion = trajectory.sample_for_deployment(1.0, arm_position, **options)
            np.testing.assert_allclose(motion.arm_position, 0.5)
            np.testing.assert_allclose(motion.arm_velocity, 2.0)
            np.testing.assert_allclose(motion.wrench_prediction, 4.0)

            transition = trajectory.sample_for_deployment(1.75, arm_position, **options)
            np.testing.assert_allclose(
                transition.base_command, (-0.3875, -0.01, 0.0)
            )

            terminal = trajectory.sample_for_deployment(2.0, arm_position, **options)
            np.testing.assert_allclose(terminal.arm_velocity, 0.0)
            np.testing.assert_allclose(terminal.base_command, (-0.25, 0.0, 0.0))

            wrench_only = trajectory.sample_for_deployment(
                0.25, arm_position, hold_arm=True
            )
            np.testing.assert_allclose(wrench_only.arm_position, arm_position)
            np.testing.assert_allclose(wrench_only.arm_effort, 0.0)
            np.testing.assert_allclose(wrench_only.wrench_prediction, 4.0)

            arm_only = trajectory.sample_for_deployment(
                0.25, arm_position, zero_wrench=True
            )
            np.testing.assert_allclose(arm_only.arm_position, 0.5625)
            np.testing.assert_allclose(arm_only.wrench_prediction, 0.0)


if __name__ == "__main__":
    unittest.main()
