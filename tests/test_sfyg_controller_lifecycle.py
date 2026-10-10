"""Exercise SFYG replay lifecycle logs without the native SDK or ONNX models."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


CONTROLLERS = Path(__file__).resolve().parents[1] / "controllers"
PACKAGE_NAME = "sfyg_lifecycle_test_controllers"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CONTROLLERS)]
sys.modules[PACKAGE_NAME] = package

for module_name, filename in (
    ("sfyg_contract", "sfyg_contract.py"),
    ("SFYGController", "SFYGController.py"),
):
    name = f"{PACKAGE_NAME}.{module_name}"
    spec = importlib.util.spec_from_file_location(name, CONTROLLERS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

controller_module = sys.modules[f"{PACKAGE_NAME}.SFYGController"]
contract_module = sys.modules[f"{PACKAGE_NAME}.sfyg_contract"]


class SfygControllerLifecycleTest(unittest.TestCase):
    def _controller(self):
        controller = controller_module.SFYGController.__new__(
            controller_module.SFYGController
        )
        q = np.zeros(18)
        dq = np.zeros(18)
        quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
        gyro = np.zeros(3)
        snapshot = lambda: (q.copy(), dq.copy(), quaternion.copy(), gyro.copy())
        controller._wait_for_state = snapshot
        controller._snapshot = snapshot
        controller._move_to_default = lambda _q: None
        controller._publish_policy_targets = lambda _targets, _solution: None
        controller._publish = lambda *args, **kwargs: None
        controller._infer = lambda _proprio, _solution: np.zeros(10)
        controller._safe_hold_solution = lambda _q: self._solution()
        controller.config = types.SimpleNamespace(
            default_q=q, kd=np.ones(18), loop_frequency=200.0, decimation=1
        )
        controller.start_controller = True
        controller.arm_test = None
        controller.last_action = np.zeros(10)
        controller.ocs2_start_delay = 0.02
        controller.ocs2_terminal_command = None
        controller.ocs2_hold_arm = False
        controller.ocs2_zero_wrench = True
        controller.trajectory = types.SimpleNamespace(
            duration=0.03,
            sample_for_deployment=lambda *args, **kwargs: self._solution(),
        )
        return controller

    @staticmethod
    def _solution():
        return contract_module.Ocs2Solution(
            time=0.0,
            arm_position=np.zeros(6),
            arm_velocity=np.zeros(6),
            arm_effort=np.zeros(6),
            base_command=np.zeros(3),
            wrench_prediction=np.zeros((5, 6)),
        )

    def test_start_complete_and_normal_exit_are_logged_once(self):
        output = io.StringIO()
        with patch.object(controller_module, "build_proprio_observation", return_value=np.zeros(42)), \
             patch.object(controller_module, "compose_joint_targets", return_value=np.zeros(18)), \
             contextlib.redirect_stdout(output):
            completed = self._controller().run(duration=0.12)

        self.assertTrue(completed)
        log = output.getvalue()
        self.assertEqual(log.count("OCS2 trajectory started"), 1)
        self.assertEqual(log.count("OCS2 trajectory completed"), 1)
        self.assertIn("SFYG shutdown: damping command published", log)
        self.assertIn("SFYG control loop exited normally", log)
        self.assertIn("trajectory_completed=True", log)

    def test_early_bounded_exit_is_not_reported_as_trajectory_complete(self):
        output = io.StringIO()
        with patch.object(controller_module, "build_proprio_observation", return_value=np.zeros(42)), \
             patch.object(controller_module, "compose_joint_targets", return_value=np.zeros(18)), \
             contextlib.redirect_stdout(output):
            completed = self._controller().run(duration=0.01)

        self.assertTrue(completed)
        log = output.getvalue()
        self.assertNotIn("OCS2 trajectory completed", log)
        self.assertIn("trajectory_completed=False", log)


if __name__ == "__main__":
    unittest.main()
