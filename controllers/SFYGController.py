#!/usr/bin/env python3
"""Paper-aligned SFYG controller: RL legs + OCS2 arm/wrench trajectory."""

import copy
import os
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .sfyg_contract import (
    Ocs2Solution,
    Ocs2Trajectory,
    SFYGPolicyConfig,
    build_proprio_observation,
    compose_joint_targets,
    compose_policy_input,
)


class SFYGController:
    """Control all 18 SFYG joints over one named Tron2 SDK channel."""

    def __init__(
        self,
        model_dir,
        robot,
        robot_type,
        start_controller=False,
        ocs2_trajectory=None,
        base_command=(0.0, 0.0, 0.0),
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required: pip install onnxruntime")
        self.robot = robot
        self.model_dir = os.path.abspath(os.path.join(model_dir, robot_type))
        self.config = SFYGPolicyConfig(os.path.join(self.model_dir, "params.yaml"))
        if self.config.robot_variant != robot_type:
            raise ValueError("SFYG params.yaml robot_variant mismatch")
        self.trajectory = (
            Ocs2Trajectory(ocs2_trajectory) if ocs2_trajectory is not None else None
        )
        self.base_command = np.asarray(base_command, dtype=np.float64)
        command_limit = np.asarray((1.0, 0.5, 1.5), dtype=np.float64)
        if self.base_command.shape != (3,) or not np.all(np.isfinite(self.base_command)):
            raise ValueError("--base-command must contain three finite values")
        if np.any(np.abs(self.base_command) > command_limit):
            raise ValueError("--base-command exceeds training ranges: |vx|<=1, |vy|<=0.5, |wz|<=1.5")
        if self.trajectory is not None and np.any(self.base_command):
            raise ValueError("--base-command cannot be combined with --ocs2-trajectory")
        self.encoder_session = self._load_session(self.config.encoder_file)
        self.policy_session = self._load_session(self.config.policy_file)
        self._validate_model_contract()

        from limxsdk import datatypes

        self._datatypes = datatypes
        self.state = datatypes.RobotState()
        self.imu = datatypes.ImuData()
        self._state_stamp = None
        self._imu_stamp = None
        self._state_error = None
        self._state_count = 0
        self._imu_count = 0
        self.start_controller = bool(start_controller)
        self.last_action = np.zeros(10, dtype=np.float64)
        self.history = np.zeros(420, dtype=np.float32)
        self._history_initialized = False

        robot.subscribeRobotState(self._state_callback)
        robot.subscribeImuData(self._imu_callback)
        print(f"Loaded SFYG parameters from {self.config.path}")
        print("Comm: one named 18-joint Tron2 channel")
        if self.trajectory is None:
            if self.start_controller:
                print(
                    "Policy: fixed base command "
                    f"{self.base_command.tolist()}, zero wrench (no OCS2)"
                )
            else:
                print("Policy idle: static default-pose hold")
        else:
            print(f"OCS2 trajectory: {self.trajectory.path}")

    def _load_session(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SFYG model missing: {path}. Export the WholeBody model; "
                "the ordinary 48D policy is incompatible."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def _feature_size(value_info):
        shape = value_info.shape
        return shape[-1] if shape and isinstance(shape[-1], int) else None

    def _validate_model_contract(self):
        contracts = (
            ("encoder input", self.encoder_session.get_inputs(), 420),
            ("encoder output", self.encoder_session.get_outputs(), 3),
            ("policy input", self.policy_session.get_inputs(), 78),
            ("policy output", self.policy_session.get_outputs(), 10),
        )
        for name, values, expected in contracts:
            if len(values) != 1:
                raise ValueError(f"{name} must have exactly one tensor")
            actual = self._feature_size(values[0])
            if actual is not None and actual != expected:
                raise ValueError(f"{name} is {actual}D, expected {expected}D")
        print("Model contract: encoder 420->3, WholeBody policy 78->10")

    def _state_callback(self, state):
        try:
            q = self._ordered_state(state.q, getattr(state, "motor_names", None))
            dq = self._ordered_state(state.dq, getattr(state, "motor_names", None))
        except ValueError as error:
            self._state_error = str(error)
            return
        copied = copy.deepcopy(state)
        copied.q = q.tolist()
        copied.dq = dq.tolist()
        self.state = copied
        self._state_stamp = time.monotonic()
        self._state_count += 1

    def _ordered_state(self, values, names):
        values = np.asarray(values, dtype=np.float64)
        if names:
            index = {name: i for i, name in enumerate(names)}
            if all(name in index for name in self.config.wire_joint_names):
                values = np.asarray(
                    [values[index[name]] for name in self.config.wire_joint_names]
                )
        if values.shape != (18,) or not np.all(np.isfinite(values)):
            raise ValueError("RobotState must contain 18 finite SFYG joints")
        return values

    def _imu_callback(self, imu):
        if len(imu.quat) < 4 or len(imu.gyro) < 3:
            return
        self.imu = copy.deepcopy(imu)
        self._imu_stamp = time.monotonic()
        self._imu_count += 1

    def _snapshot(self):
        now = time.monotonic()
        if self._state_stamp is None or self._imu_stamp is None:
            raise RuntimeError(
                f"waiting for SFYG streams: state={self._state_count}, imu={self._imu_count}, "
                f"last_error={self._state_error}"
            )
        if now - self._state_stamp > self.config.state_timeout:
            raise RuntimeError("SFYG RobotState is stale")
        if now - self._imu_stamp > self.config.state_timeout:
            raise RuntimeError("SFYG IMU is stale")
        return (
            np.asarray(self.state.q, dtype=np.float64),
            np.asarray(self.state.dq, dtype=np.float64),
            np.asarray(self.imu.quat[:4], dtype=np.float64),
            np.asarray(self.imu.gyro[:3], dtype=np.float64),
        )

    def _make_command(self, q, dq=None, tau=None, kp=None, kd=None):
        command = self._datatypes.RobotCmd()
        command.mode = [0.0] * 18
        command.q = np.asarray(q, dtype=np.float64).tolist()
        command.dq = np.zeros(18).tolist() if dq is None else np.asarray(dq).tolist()
        command.tau = np.zeros(18).tolist() if tau is None else np.asarray(tau).tolist()
        command.Kp = self.config.kp.tolist() if kp is None else np.asarray(kp).tolist()
        command.Kd = self.config.kd.tolist() if kd is None else np.asarray(kd).tolist()
        command.motor_names = list(self.config.wire_joint_names)
        return command

    def _publish(self, q, dq=None, tau=None, kp=None, kd=None):
        self.robot.publishRobotCmd(self._make_command(q, dq, tau, kp, kd))

    def _wait_for_state(self):
        deadline = time.monotonic() + self.config.state_wait_timeout
        while time.monotonic() < deadline:
            try:
                return self._snapshot()
            except RuntimeError:
                time.sleep(0.01)
        raise RuntimeError("timed out waiting for 18-joint SFYG state and IMU")

    def _move_to_default(self, start_q):
        # The SFYG MuJoCo variant now starts in the exact training default
        # pose and holds physics until this controller publishes.  Replaying a
        # three-second static-PD stand phase would release gravity while the
        # balance policy is still disabled.  When the measured pose is already
        # aligned, send one complete command and enter policy inference on the
        # next control tick.  A displaced/real robot still gets the smooth
        # transition below.
        if self.start_controller and np.max(
            np.abs(np.asarray(start_q) - self.config.default_q)
        ) <= 0.05:
            self._publish(self.config.default_q)
            print("Initial pose already aligned; entering balance policy immediately")
            return
        started = time.monotonic()
        period = 1.0 / self.config.loop_frequency
        next_tick = started
        while True:
            ratio = min((time.monotonic() - started) / self.config.stand_duration, 1.0)
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            self._publish(start_q * (1.0 - smooth) + self.config.default_q * smooth)
            if ratio >= 1.0:
                return
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    @staticmethod
    def _onnx_input(session, vector):
        value = np.asarray(vector, dtype=np.float32)
        info = session.get_inputs()[0]
        if len(info.shape) == 2:
            value = value.reshape(1, -1)
        return {info.name: value}

    def _infer(self, proprio, solution):
        if not self._history_initialized:
            self.history[:] = np.tile(proprio, self.config.history_length)
            self._history_initialized = True
        else:
            self.history[:-42] = self.history[42:]
            self.history[-42:] = proprio
        encoder = self.encoder_session.run(
            None, self._onnx_input(self.encoder_session, self.history)
        )[0]
        encoder = np.asarray(encoder, dtype=np.float32).reshape(-1)
        policy_input = compose_policy_input(
            self.config,
            encoder,
            proprio,
            solution.wrench_prediction,
            solution.base_command.astype(np.float32),
        )
        action = self.policy_session.run(
            None, self._onnx_input(self.policy_session, policy_input)
        )[0]
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (10,) or not np.all(np.isfinite(action)):
            raise RuntimeError("WholeBody policy returned an invalid action")
        return np.clip(action, -self.config.action_clip, self.config.action_clip)

    def _safe_hold_solution(self, q):
        return Ocs2Solution(
            time=0.0,
            arm_position=q[10:16].copy(),
            arm_velocity=np.zeros(6),
            arm_effort=np.zeros(6),
            base_command=self.base_command.copy(),
            wrench_prediction=np.zeros((5, 6)),
        )

    def _publish_policy_targets(self, targets, solution):
        dq = np.zeros(18)
        tau = np.zeros(18)
        dq[10:16] = solution.arm_velocity
        tau[10:16] = solution.arm_effort
        self._publish(targets, dq=dq, tau=tau)

    def run(self, duration=0.0):
        try:
            initial_q, _, _, _ = self._wait_for_state()
            self._move_to_default(initial_q)
            if not self.start_controller:
                print("SFYG is holding the default pose; use --start-controller to enable balance policy")
            started = time.monotonic()
            next_tick = started
            control_period = 1.0 / self.config.loop_frequency
            loop_count = 0
            last_targets = self.config.default_q.copy()
            last_solution = self._safe_hold_solution(last_targets)
            while duration <= 0.0 or time.monotonic() - started < duration:
                q, dq, quaternion, gyro = self._snapshot()
                if self.start_controller and loop_count % self.config.decimation == 0:
                    elapsed = time.monotonic() - started
                    if self.trajectory is None:
                        last_solution = self._safe_hold_solution(last_targets)
                    else:
                        last_solution = self.trajectory.sample(elapsed)
                    proprio = build_proprio_observation(
                        self.config,
                        q,
                        dq,
                        gyro,
                        quaternion,
                        self.last_action,
                        last_solution.base_command,
                        elapsed,
                    )
                    self.last_action = self._infer(proprio, last_solution)
                    last_targets = compose_joint_targets(
                        self.config, self.last_action, last_solution
                    )
                self._publish_policy_targets(last_targets, last_solution)
                loop_count += 1
                next_tick += control_period
                if next_tick < time.monotonic() - control_period:
                    next_tick = time.monotonic()
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            print("SFYG controller interrupted")
        finally:
            try:
                q, _, _, _ = self._snapshot()
                self._publish(q, kp=np.zeros(18), kd=self.config.kd)
            except RuntimeError:
                pass
