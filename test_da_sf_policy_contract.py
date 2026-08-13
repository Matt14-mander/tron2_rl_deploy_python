#!/usr/bin/env python3
"""Offline contract tests for controllers/DASFController.py."""

import importlib.util
import math
import pathlib
import threading
import types
import unittest

import numpy as np


ROOT_PATH = pathlib.Path(__file__).parent
MODULE_PATH = ROOT_PATH / "controllers" / "DASFController.py"
CONFIG_PATH = ROOT_PATH / "controllers" / "model" / "DA_SF_TRON2A" / "params.yaml"
SPEC = importlib.util.spec_from_file_location("DASFController", MODULE_PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class DASFPolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = controller.DASFPolicyConfig(CONFIG_PATH)

    def test_projected_gravity_uses_wxyz_quaternion(self):
        np.testing.assert_allclose(
            controller.projected_gravity_from_wxyz([1.0, 0.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1.0e-7,
        )
        half_sqrt = math.sqrt(0.5)
        np.testing.assert_allclose(
            controller.projected_gravity_from_wxyz(
                [half_sqrt, half_sqrt, 0.0, 0.0]
            ),
            [0.0, -1.0, 0.0],
            atol=1.0e-7,
        )

    def test_pointfoot_config_matches_deployment_contract(self):
        self.assertEqual(self.config.robot_variant, "DA_SF_TRON2A")
        self.assertEqual(self.config.policy_frequency, 50.0)
        self.assertEqual(self.config.stand_control_frequency, 200.0)
        self.assertEqual(self.config.phase_period, 1.0)
        self.assertEqual(self.config.max_command[0], 1.0)
        self.assertEqual(self.config.encoder_file, "encoder.onnx")
        self.assertEqual(self.config.policy_file, "policy.onnx")

    def test_default_pose_builds_expected_standing_observation(self):
        observation = controller.build_policy_observation(
            self.config,
            full_q=self.config.default_q,
            full_dq=np.zeros(self.config.num_joints),
            gyro=np.zeros(3),
            quaternion=[1.0, 0.0, 0.0, 0.0],
            last_action=np.zeros(self.config.num_actions),
            command=np.zeros(3),
            policy_elapsed=0.0,
        )
        self.assertEqual(observation.shape, (self.config.obs_size,))
        np.testing.assert_allclose(observation[:3], np.zeros(3))
        np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(observation[6:66], np.zeros(60))
        np.testing.assert_allclose(observation[66:68], np.zeros(2))

    def test_moving_command_enables_one_second_phase(self):
        observation = controller.build_policy_observation(
            self.config,
            full_q=self.config.default_q,
            full_dq=np.zeros(self.config.num_joints),
            gyro=np.zeros(3),
            quaternion=[1.0, 0.0, 0.0, 0.0],
            last_action=np.zeros(self.config.num_actions),
            command=[0.2, 0.0, 0.0],
            policy_elapsed=0.25,
        )
        np.testing.assert_allclose(observation[-2:], [1.0, 0.0], atol=1.0e-7)

    def test_each_policy_action_reaches_only_its_mapped_joint(self):
        locked_indices = sorted(
            set(range(self.config.num_joints)) - set(self.config.policy_to_full)
        )
        self.assertEqual(locked_indices, [14, 15, 21, 22, 24, 25])

        for action_index, full_index in enumerate(self.config.policy_to_full):
            actions = np.zeros(self.config.num_actions)
            actions[action_index] = 1.0
            targets = controller.actions_to_full_targets(self.config, actions)
            delta = targets - self.config.default_q
            nonzero = np.flatnonzero(np.abs(delta) > 1.0e-12)
            np.testing.assert_array_equal(nonzero, [full_index])
            self.assertAlmostEqual(
                delta[full_index], self.config.action_scales[action_index]
            )

    def test_model_and_gain_dimensions_are_consistent(self):
        self.assertEqual(self.config.history_size, 680)
        self.assertEqual(self.config.policy_input_size, 74)
        self.assertEqual(self.config.action_scales.shape, (20,))
        self.assertEqual(self.config.kp.shape, (26,))
        self.assertEqual(self.config.kd.shape, (26,))
        self.assertAlmostEqual(self.config.default_q[11], math.radians(165.0))
        self.assertAlmostEqual(self.config.default_q[18], math.radians(-165.0))

    def test_hardware_joint_zero_maps_to_simulation_zero(self):
        hardware_q = np.zeros(self.config.num_joints)
        hardware_dq = np.arange(self.config.num_joints, dtype=np.float64)
        hardware_q[2] = -math.pi
        hardware_q[7] = math.pi
        hardware_q[[3, 4, 8, 9]] = [0.2, -0.3, 0.4, -0.5]

        policy_q, policy_dq = controller.hardware_to_policy_state(
            self.config, hardware_q, hardware_dq
        )

        np.testing.assert_allclose(policy_q[[2, 7]], [0.0, 0.0])
        np.testing.assert_allclose(policy_q[[3, 4, 8, 9]], [-0.2, 0.3, -0.4, 0.5])
        np.testing.assert_allclose(
            policy_dq[[3, 4, 8, 9]], -hardware_dq[[3, 4, 8, 9]]
        )

    def test_hardware_joint_transform_round_trip(self):
        policy_targets = np.linspace(-0.8, 0.8, self.config.num_joints)
        hardware_targets = controller.policy_targets_to_hardware(
            self.config, policy_targets
        )
        recovered_targets, _ = controller.hardware_to_policy_state(
            self.config, hardware_targets, np.zeros(self.config.num_joints)
        )

        np.testing.assert_allclose(recovered_targets, policy_targets)
        self.assertAlmostEqual(hardware_targets[2], policy_targets[2] - math.pi)
        self.assertAlmostEqual(hardware_targets[7], policy_targets[7] + math.pi)
        np.testing.assert_allclose(
            hardware_targets[[3, 4, 8, 9]], -policy_targets[[3, 4, 8, 9]]
        )

    def test_hardware_transform_detection_can_be_overridden(self):
        self.assertTrue(
            controller.hardware_joint_transform_enabled(
                {"MROS_IP_LIST": "10.192.1.x"}
            )
        )
        self.assertFalse(
            controller.hardware_joint_transform_enabled(
                {
                    "MROS_IP_LIST": "10.192.1.x",
                    "DA_SF_HARDWARE_JOINT_TRANSFORM": "0",
                }
            )
        )

    def test_controller_applies_transform_to_state_and_targets(self):
        class FakePublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        policy = controller.DASFController.__new__(controller.DASFController)
        policy.config = self.config
        policy.state_timeout = 1.0
        policy.use_hardware_joint_transform = True
        policy._lock = threading.Lock()
        expected_policy_q = self.config.default_q.copy()
        expected_policy_q[[2, 3, 4, 7, 8, 9]] += [0.1, 0.2, -0.3, -0.4, 0.5, -0.6]
        expected_policy_dq = np.zeros(self.config.num_joints)
        expected_policy_dq[[2, 3, 4, 7, 8, 9]] = [0.7, -0.8, 0.9, -1.0, 1.1, -1.2]
        hardware_q = controller.policy_targets_to_hardware(
            self.config, expected_policy_q
        )
        hardware_dq = (
            self.config.hardware_joint_direction * expected_policy_dq
        )
        policy._lower_q = hardware_q[: self.config.num_lower].copy()
        policy._lower_dq = hardware_dq[: self.config.num_lower].copy()
        policy._upper_q = hardware_q[self.config.num_lower :].copy()
        policy._upper_dq = hardware_dq[self.config.num_lower :].copy()
        policy._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        policy._gyro = np.zeros(3)
        policy._lower_stamp = policy._upper_stamp = policy._imu_stamp = (
            controller.time.monotonic()
        )
        policy.pub_lower = FakePublisher()
        policy.pub_upper = FakePublisher()

        policy_q, policy_dq, _, _ = policy._snapshot()
        np.testing.assert_allclose(policy_q, expected_policy_q)
        np.testing.assert_allclose(policy_dq, expected_policy_dq)

        observation = controller.build_policy_observation(
            self.config,
            policy_q,
            policy_dq,
            gyro=np.zeros(3),
            quaternion=[1.0, 0.0, 0.0, 0.0],
            last_action=np.zeros(self.config.num_actions),
            command=np.zeros(3),
            policy_elapsed=0.0,
        )
        expected_position_observation = (
            expected_policy_q[self.config.policy_to_full]
            - self.config.default_q[self.config.policy_to_full]
        ) * self.config.dof_pos_scale
        expected_velocity_observation = (
            expected_policy_dq[self.config.policy_to_full]
            * self.config.dof_vel_scale
        )
        np.testing.assert_allclose(
            observation[6 : 6 + self.config.num_actions],
            expected_position_observation,
        )
        np.testing.assert_allclose(
            observation[
                6 + self.config.num_actions : 6 + 2 * self.config.num_actions
            ],
            expected_velocity_observation,
        )

        actions = np.linspace(-0.5, 0.5, self.config.num_actions)
        policy_targets = controller.actions_to_full_targets(self.config, actions)
        expected_hardware_targets = controller.policy_targets_to_hardware(
            self.config, policy_targets
        )
        policy._publish_targets(policy_targets)
        published_targets = np.concatenate(
            (policy.pub_lower.messages[0].q, policy.pub_upper.messages[0].q)
        )
        np.testing.assert_allclose(published_targets, expected_hardware_targets)
        np.testing.assert_allclose(
            policy.pub_lower.messages[0].v,
            np.zeros(self.config.num_lower),
        )
        np.testing.assert_allclose(
            policy.pub_lower.messages[0].tau,
            np.zeros(self.config.num_lower),
        )

    def test_robot_joystick_starts_scales_resets_and_stops_policy(self):
        policy = controller.DASFController.__new__(controller.DASFController)
        policy.config = self.config
        policy._control_lock = threading.Lock()
        policy.command = np.zeros(3, dtype=np.float64)
        policy.start_controller = False
        policy.mode = "STAND"
        policy._history_reset_requested = False
        policy._walk_start_time = 0.0
        policy._last_r1 = 0
        policy.joy_msg_count = 0
        policy.last_joy_time = 0.0

        start_message = types.SimpleNamespace(
            axes=[0.2, 0.4, -0.5],
            buttons=[0, 0, 0, 1, 1, 0],
        )
        policy.sensor_joy_callback(start_message)
        self.assertTrue(policy.start_controller)
        self.assertEqual(policy.mode, "WALK")
        self.assertTrue(policy._history_reset_requested)
        np.testing.assert_allclose(
            policy.command,
            np.array([0.4, 0.2, -0.5]) * self.config.max_command,
        )
        self.assertEqual(policy.joy_msg_count, 1)
        self.assertTrue(policy.joystick_alive())

        policy._history_reset_requested = False
        clear_message = types.SimpleNamespace(
            axes=[0.0, 0.0, 0.0],
            buttons=[0, 0, 0, 0, 0, 1],
        )
        policy.sensor_joy_callback(clear_message)
        np.testing.assert_allclose(policy.command, np.zeros(3))
        self.assertTrue(policy._history_reset_requested)

        stop_message = types.SimpleNamespace(
            axes=[0.0, 0.0, 0.0],
            buttons=[0, 0, 1, 0, 1, 0],
        )
        policy.sensor_joy_callback(stop_message)
        self.assertFalse(policy.start_controller)
        self.assertEqual(policy.mode, "IDLE")
        np.testing.assert_allclose(policy.command, np.zeros(3))

    def test_direct_f710_uses_xinput_axes_and_buttons(self):
        class FakeJoystick:
            @staticmethod
            def get_numaxes():
                return 6

            @staticmethod
            def get_numbuttons():
                return 18

            @staticmethod
            def get_axis(index):
                return [0.25, -0.5, 0.0, -0.4, 0.0, 0.0][index]

            @staticmethod
            def get_button(index):
                return [0, 0, 0, 1, 1, 0][index]

        policy = controller.DASFController.__new__(controller.DASFController)
        policy.config = self.config
        policy._control_lock = threading.Lock()
        policy.command = np.zeros(3, dtype=np.float64)
        policy.start_controller = False
        policy.mode = "STAND"
        policy._history_reset_requested = False
        policy._walk_start_time = 0.0
        policy._pygame_last_r1 = 0
        policy._pygame_enabled = True
        policy._pygame_joystick = FakeJoystick()
        policy.joy_msg_count = 0
        policy.last_joy_time = 0.0

        original_pygame = controller.pygame
        controller.pygame = types.SimpleNamespace(
            event=types.SimpleNamespace(pump=lambda: None)
        )
        try:
            policy._poll_pygame_joystick()
        finally:
            controller.pygame = original_pygame

        self.assertTrue(policy.start_controller)
        self.assertEqual(policy.mode, "WALK")
        np.testing.assert_allclose(
            policy.command,
            np.array([0.5, -0.25, 0.4]) * self.config.max_command,
        )
        self.assertEqual(policy.joy_msg_count, 1)


if __name__ == "__main__":
    unittest.main()