"""Pure numerical contracts for SFYG whole-body deployment."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

WRENCH_TIMES = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8), dtype=np.float64)
WRENCH_SCALE = np.asarray((0.01, 0.01, 0.01, 0.05, 0.05, 0.05), dtype=np.float64)


@dataclass(frozen=True)
class Ocs2Solution:
    time: float
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    arm_effort: np.ndarray
    base_command: np.ndarray
    wrench_prediction: np.ndarray


def validate_ocs2_solution(solution):
    expected = {
        "arm_position": (6,),
        "arm_velocity": (6,),
        "arm_effort": (6,),
        "base_command": (3,),
        "wrench_prediction": (5, 6),
    }
    if not np.isfinite(solution.time):
        raise ValueError("OCS2 solution time must be finite")
    for name, shape in expected.items():
        value = np.asarray(getattr(solution, name))
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"OCS2 {name} must be finite with shape {shape}")


def trajectory_columns():
    names = ["time"]
    names.extend(f"arm_q{i}" for i in range(1, 7))
    names.extend(f"arm_dq{i}" for i in range(1, 7))
    names.extend(f"arm_tau{i}" for i in range(1, 7))
    names.extend(("base_vx", "base_vy", "base_wz"))
    for sample in range(5):
        names.extend(
            f"w{sample}_{component}"
            for component in ("fx", "fy", "fz", "tx", "ty", "tz")
        )
    return tuple(names)


TRAJECTORY_COLUMNS = trajectory_columns()

# SFYG_TRON2A URDF arm limits. Keep a margin so diagnostic motions cannot
# reproduce the out-of-range arm4 target seen in an older OCS2 export.
ARM_TEST_LIMITS = {
    "arm1": (-2.6179938, 2.6179938),
    "arm2": (0.0, 3.1415926),
    "arm3": (-2.9670597, 0.0),
    "arm4": (-1.5533430, 1.5533430),
    "arm5": (-1.5533430, 1.5533430),
    "arm6": (-2.0943951, 2.0943951),
}

# Mirror the OCS2 export checks when loading a trajectory on the deployment
# side, so older CSVs cannot bypass the corrected exporter's safety limits.
ARM_TRAJECTORY_POSITION_MARGIN = 0.05
ARM_TRAJECTORY_VELOCITY_LIMIT = 5.0
ARM_TRAJECTORY_EFFORT_LIMIT = 100.0
BASE_COMMAND_LIMITS = np.asarray((1.0, 0.5, 1.5), dtype=np.float64)
TRAJECTORY_LIMIT_TOLERANCE = 1.0e-6


class Ocs2Trajectory:
    """Validated interpolation of the WholeBody Lab 52-column export."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"OCS2 trajectory not found: {self.path}")
        with open(self.path, encoding="utf-8") as stream:
            header = tuple(part.strip() for part in stream.readline().split(","))
        if header != TRAJECTORY_COLUMNS:
            raise ValueError("OCS2 trajectory does not use the 52-column v1 contract")
        self.data = np.loadtxt(
            self.path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64
        )
        if self.data.shape[1] != 52 or self.data.shape[0] < 2:
            raise ValueError("OCS2 trajectory must have shape (N>=2, 52)")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("OCS2 trajectory contains NaN or Inf")
        if abs(self.data[0, 0]) > 1.0e-9 or np.any(np.diff(self.data[:, 0]) <= 0):
            raise ValueError("OCS2 trajectory must start at zero and increase strictly")
        self._validate_limits()
        self._terminal_hold_start = None
        if (
            np.allclose(self.data[-2, 1:7], self.data[-1, 1:7], rtol=0, atol=1e-9)
            and np.allclose(self.data[-1, 7:13], 0.0, rtol=0, atol=1e-9)
        ):
            self._terminal_hold_start = float(self.data[-2, 0])

    def _validate_limits(self):
        """Reject unsafe legacy exports before any command reaches the robot."""
        tolerance = TRAJECTORY_LIMIT_TOLERANCE
        for index, (name, (lower, upper)) in enumerate(ARM_TEST_LIMITS.items()):
            for field, values, lower_bound, upper_bound in (
                (
                    "position",
                    self.data[:, 1 + index],
                    lower + ARM_TRAJECTORY_POSITION_MARGIN,
                    upper - ARM_TRAJECTORY_POSITION_MARGIN,
                ),
                (
                    "velocity",
                    self.data[:, 7 + index],
                    -ARM_TRAJECTORY_VELOCITY_LIMIT,
                    ARM_TRAJECTORY_VELOCITY_LIMIT,
                ),
                (
                    "effort",
                    self.data[:, 13 + index],
                    -ARM_TRAJECTORY_EFFORT_LIMIT,
                    ARM_TRAJECTORY_EFFORT_LIMIT,
                ),
            ):
                invalid = np.flatnonzero(
                    (values < lower_bound - tolerance) | (values > upper_bound + tolerance)
                )
                if invalid.size:
                    row = int(invalid[0])
                    raise ValueError(
                        f"Unsafe OCS2 trajectory {self.path}: {name} {field} "
                        f"at t={self.data[row, 0]:.3f}s is {values[row]:.6f}; "
                        f"allowed [{lower_bound:.6f}, {upper_bound:.6f}]"
                    )
        for index, name in enumerate(("base_vx", "base_vy", "base_wz")):
            values = self.data[:, 19 + index]
            limit = BASE_COMMAND_LIMITS[index]
            invalid = np.flatnonzero(np.abs(values) > limit + tolerance)
            if invalid.size:
                row = int(invalid[0])
                raise ValueError(
                    f"Unsafe OCS2 trajectory {self.path}: {name} "
                    f"at t={self.data[row, 0]:.3f}s is {values[row]:.6f}; "
                    f"allowed [{-limit:.6f}, {limit:.6f}]"
                )

    @property
    def duration(self):
        return float(self.data[-1, 0])

    def sample(self, time_s):
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("trajectory time must be finite and non-negative")
        times = self.data[:, 0]
        if time_s >= times[-1]:
            row = self.data[-1].copy()
            row[7:13] = 0.0
            row[19:22] = 0.0
        else:
            upper = int(np.searchsorted(times, time_s, side="right"))
            lower = max(0, upper - 1)
            alpha = (time_s - times[lower]) / (times[upper] - times[lower])
            row = (1.0 - alpha) * self.data[lower] + alpha * self.data[upper]
        if self._terminal_hold_start is not None and time_s >= self._terminal_hold_start:
            # The export repeats its last arm position for the terminal hold.
            # Interpolating a nonzero terminal MPC velocity into that fixed
            # position would keep driving the arm through the hold interval.
            row[7:13] = 0.0
        solution = Ocs2Solution(
            time=float(time_s),
            arm_position=row[1:7].copy(),
            arm_velocity=row[7:13].copy(),
            arm_effort=row[13:19].copy(),
            base_command=row[19:22].copy(),
            wrench_prediction=row[22:52].reshape(5, 6).copy(),
        )
        validate_ocs2_solution(solution)
        return solution

    def sample_for_deployment(
        self,
        elapsed,
        arm_position,
        start_delay=0.0,
        terminal_command=None,
        terminal_transition=0.5,
        hold_arm=False,
        zero_wrench=False,
    ):
        """Replay an OCS2 plan with optional startup, terminal, and diagnostic phases."""
        playback_time = max(0.0, elapsed - start_delay)
        solution = self.sample(playback_time)

        if elapsed < start_delay:
            # Establish policy history before moving the arm. Keep the first
            # planned base command, which may already be a stable walking gait.
            solution = Ocs2Solution(
                time=solution.time,
                arm_position=np.asarray(arm_position, dtype=np.float64).copy(),
                arm_velocity=np.zeros(6),
                arm_effort=np.zeros(6),
                base_command=solution.base_command,
                wrench_prediction=np.zeros((5, 6)),
            )
        elif terminal_command is not None:
            # Match the Isaac Lab rollout's smooth transition near the end.
            transition_start = max(0.0, self.duration - terminal_transition)
            alpha = np.clip(
                (playback_time - transition_start) / terminal_transition, 0.0, 1.0
            )
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            solution = Ocs2Solution(
                time=solution.time,
                arm_position=solution.arm_position,
                arm_velocity=solution.arm_velocity,
                arm_effort=solution.arm_effort,
                base_command=(1.0 - smooth) * solution.base_command
                + smooth * np.asarray(terminal_command, dtype=np.float64),
                wrench_prediction=solution.wrench_prediction,
            )

        if hold_arm or zero_wrench:
            solution = Ocs2Solution(
                time=solution.time,
                arm_position=(
                    np.asarray(arm_position, dtype=np.float64).copy()
                    if hold_arm else solution.arm_position
                ),
                arm_velocity=np.zeros(6) if hold_arm else solution.arm_velocity,
                arm_effort=np.zeros(6) if hold_arm else solution.arm_effort,
                base_command=solution.base_command,
                wrench_prediction=(
                    np.zeros((5, 6)) if zero_wrench else solution.wrench_prediction
                ),
            )
        validate_ocs2_solution(solution)
        return solution


class JointSpaceArmTest:
    """Small, repeatable, one- or two-joint round trip for sim2sim diagnosis."""

    def __init__(
        self,
        default_arm_position,
        joint=None,
        delta=None,
        start_delay=3.0,
        move_duration=2.0,
        hold_duration=1.0,
        offsets=None,
    ):
        self.default_arm_position = np.asarray(default_arm_position, dtype=np.float64)
        if self.default_arm_position.shape != (6,) or not np.all(
            np.isfinite(self.default_arm_position)
        ):
            raise ValueError("arm test requires six finite default joint positions")
        single_requested = joint is not None or delta is not None
        if single_requested == (offsets is not None):
            raise ValueError("choose either one arm joint/delta or six arm offsets")
        if single_requested:
            if joint not in ARM_TEST_LIMITS or delta is None:
                raise ValueError("arm test requires an arm1-arm6 joint and delta")
            self.delta = float(delta)
            if not np.isfinite(self.delta) or not 1.0e-9 < abs(self.delta) <= 0.15:
                raise ValueError("arm test delta must be nonzero and at most 0.15 rad")
            self.offsets = np.zeros(6, dtype=np.float64)
            self.offsets[int(joint[-1]) - 1] = self.delta
        else:
            self.delta = None
            self.offsets = np.asarray(offsets, dtype=np.float64)
            if self.offsets.shape != (6,) or not np.all(np.isfinite(self.offsets)):
                raise ValueError("multi-joint arm test requires six finite offsets")
            active = np.flatnonzero(np.abs(self.offsets) > 1.0e-9)
            if len(active) != 2 or np.max(np.abs(self.offsets)) > 0.10 or (
                np.linalg.norm(self.offsets) > 0.12
            ):
                raise ValueError(
                    "multi-joint arm test requires exactly two nonzero offsets, "
                    "each <=0.10 rad and combined norm <=0.12 rad"
                )
        self.active_indices = np.flatnonzero(np.abs(self.offsets) > 1.0e-9)
        self.joint = joint if single_requested else "multi"
        self.joint_index = int(self.active_indices[0]) if single_requested else None
        self.description = ", ".join(
            f"arm{i + 1}={self.offsets[i]:+.3f} rad" for i in self.active_indices
        )
        self.start_delay = float(start_delay)
        self.move_duration = float(move_duration)
        self.hold_duration = float(hold_duration)
        if not np.isfinite(self.start_delay) or self.start_delay < 2.0:
            raise ValueError("arm test start delay must be at least 2 s")
        if not np.isfinite(self.move_duration) or self.move_duration < 1.5:
            raise ValueError("arm test move duration must be at least 1.5 s")
        if not np.isfinite(self.hold_duration) or self.hold_duration < 0.5:
            raise ValueError("arm test hold duration must be at least 0.5 s")
        motion_norm = np.linalg.norm(self.offsets)
        if 1.875 * motion_norm / self.move_duration > 0.25 or (
            5.773503 * motion_norm / self.move_duration**2 > 0.5
        ):
            raise ValueError("arm test motion exceeds 0.25 rad/s or 0.5 rad/s^2")
        for i in self.active_indices:
            name = f"arm{i + 1}"
            lower, upper = ARM_TEST_LIMITS[name]
            origin = self.default_arm_position[i]
            if not lower + 0.1 <= origin <= upper - 0.1 or not (
                lower + 0.1 <= origin + self.offsets[i] <= upper - 0.1
            ):
                raise ValueError(f"arm test target exceeds {name} limits with 0.1 rad margin")

    @staticmethod
    def _blend(elapsed, duration):
        u = np.clip(elapsed / duration, 0.0, 1.0)
        position = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        velocity = 30.0 * u**2 * (1.0 - u) ** 2 / duration
        return position, velocity

    def phase(self, elapsed):
        if elapsed < self.start_delay:
            return "warmup"
        if elapsed < self.start_delay + self.move_duration:
            return "outbound"
        if elapsed < self.start_delay + self.move_duration + self.hold_duration:
            return "hold"
        if elapsed < self.start_delay + 2.0 * self.move_duration + self.hold_duration:
            return "return"
        return "complete"

    def sample(self, elapsed, base_command):
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("arm test time must be finite and non-negative")
        command = np.asarray(base_command, dtype=np.float64)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("arm test base command must contain three finite values")
        position = self.default_arm_position.copy()
        velocity = np.zeros(6, dtype=np.float64)
        phase = self.phase(elapsed)
        if phase == "outbound":
            blend, blend_rate = self._blend(elapsed - self.start_delay, self.move_duration)
            position += self.offsets * blend
            velocity = self.offsets * blend_rate
        elif phase == "hold":
            position += self.offsets
        elif phase == "return":
            return_start = self.start_delay + self.move_duration + self.hold_duration
            blend, blend_rate = self._blend(elapsed - return_start, self.move_duration)
            position += self.offsets * (1.0 - blend)
            velocity = -self.offsets * blend_rate
        solution = Ocs2Solution(
            time=float(elapsed),
            arm_position=position,
            arm_velocity=velocity,
            arm_effort=np.zeros(6),
            base_command=command.copy(),
            wrench_prediction=np.zeros((5, 6)),
        )
        validate_ocs2_solution(solution)
        return solution


def arm_test_tracking_ok(error, velocity, tilt_sin, baseline_error=None):
    """Allow static gravity sag but reject new tracking error or fast motion."""
    error = np.asarray(error, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if error.shape != (6,) or velocity.shape != (6,) or not np.all(
        np.isfinite(error)
    ) or not np.all(np.isfinite(velocity)) or not np.isfinite(tilt_sin):
        return False
    if baseline_error is None:
        return (
            np.max(np.abs(error)) <= 0.2
            and np.max(np.abs(velocity)) <= 0.5
            and tilt_sin <= 0.3
        )
    baseline = np.asarray(baseline_error, dtype=np.float64)
    if baseline.shape != (6,) or not np.all(np.isfinite(baseline)):
        return False
    return (
        np.max(np.abs(error)) <= 0.3
        and np.max(np.abs(error - baseline)) <= 0.12
        and np.max(np.abs(velocity)) <= 1.0
        and tilt_sin <= 0.45
    )


class SFYGPolicyConfig:
    """Validated SFYG runtime configuration."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(self.path, encoding="utf-8") as stream:
            root = yaml.safe_load(stream)
        config = root["PointfootCfg"]
        self.robot_variant = str(config["robot_variant"])
        init = config["init_state"]
        self.joint_names = list(init["joint_names"])
        self.leg_joint_names = list(init["policy_joint_names"])
        self.arm_joint_names = list(init["arm_joint_names"])
        self.gripper_joint_names = list(init["gripper_joint_names"])
        if self.joint_names != (
            self.leg_joint_names + self.arm_joint_names + self.gripper_joint_names
        ):
            raise ValueError("SFYG joint order must be legs, arm, then gripper")
        if (len(self.joint_names), len(self.leg_joint_names)) != (18, 10):
            raise ValueError("SFYG requires 18 joints and 10 policy joints")
        self.wire_joint_names = [f"{name}_Joint" for name in self.joint_names]
        self.default_q = self._ordered(init["default_joint_angle"], self.joint_names)
        control = config["control"]
        self.kp = self._ordered(control["stiffness"], self.joint_names)
        self.kd = self._ordered(control["damping"], self.joint_names)
        self.action_scale = self._ordered(
            control["action_scale_pos"], self.leg_joint_names
        )
        self.decimation = int(control["decimation"])
        self.loop_frequency = float(config["loop_frequency"])
        self.policy_frequency = self.loop_frequency / self.decimation
        scales = config["normalization"]["obs_scales"]
        self.ang_vel_scale = float(scales["ang_vel"])
        self.dof_pos_scale = float(scales["dof_pos"])
        self.dof_vel_scale = float(scales["dof_vel"])
        clips = config["normalization"]["clip_scales"]
        self.observation_clip = abs(float(clips["clip_observations"]))
        self.action_clip = abs(float(clips["clip_actions"]))
        size = config["size"]
        self.num_actions = int(size["actions_size"])
        self.proprio_size = int(size["proprio_obs_size"])
        self.wrench_size = int(size["wrench_prediction_size"])
        self.history_length = int(size["obs_history_length"])
        self.encoder_output_size = int(size["encoder_output_size"])
        self.command_size = int(size["commands_obs_size"])
        self.history_size = self.proprio_size * self.history_length
        self.policy_input_size = (
            self.encoder_output_size
            + self.proprio_size
            + self.wrench_size
            + self.command_size
        )
        expected = (10, 42, 30, 420, 3, 3, 78)
        actual = (
            self.num_actions,
            self.proprio_size,
            self.wrench_size,
            int(size["encoder_input_size"]),
            self.encoder_output_size,
            self.command_size,
            int(size["policy_input_size"]),
        )
        if actual != expected or self.history_size != 420 or self.policy_input_size != 78:
            raise ValueError(f"SFYG model contract mismatch: {actual}")
        gait = config["gait"]
        self.gait = np.asarray(
            [gait["frequency"], gait["offset"], gait["duration"], gait["swing_height"]],
            dtype=np.float64,
        )
        self.command_threshold = float(gait["command_threshold"])
        runtime = config["runtime"]
        self.state_timeout = float(runtime["state_timeout"])
        self.state_wait_timeout = float(runtime["state_wait_timeout"])
        self.stand_duration = float(config["stand_mode"]["stand_duration"])
        self.encoder_file = "encoder.onnx"
        self.policy_file = "policy.onnx"

    @staticmethod
    def _ordered(values, names):
        if set(values) != set(names):
            raise ValueError("joint-keyed configuration does not match joint_names")
        result = np.asarray([values[name] for name in names], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("joint-keyed configuration contains NaN or Inf")
        return result


def projected_gravity_from_wxyz(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("invalid IMU quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1]
    )


def build_proprio_observation(config, q, dq, gyro, quaternion, last_action, command, elapsed):
    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if q.shape != (18,) or dq.shape != (18,) or command.shape != (3,):
        raise ValueError("SFYG state must be 18D and command must be 3D")
    # Match the training observation exactly.  ``get_gait_phase`` in the
    # Isaac Lab task advances from episode time even for a standing velocity
    # command; freezing this at [0, 1] makes the 10-frame encoder history
    # out-of-distribution.
    angle = 2.0 * math.pi * ((elapsed * config.gait[0]) % 1.0)
    phase = np.asarray((math.sin(angle), math.cos(angle)))
    observation = np.concatenate(
        (
            np.asarray(gyro) * config.ang_vel_scale,
            projected_gravity_from_wxyz(quaternion),
            (q[:10] - config.default_q[:10]) * config.dof_pos_scale,
            dq[:10] * config.dof_vel_scale,
            np.asarray(last_action),
            phase,
            config.gait,
        )
    )
    if observation.shape != (42,) or not np.all(np.isfinite(observation)):
        raise ValueError("invalid 42D SFYG proprioceptive observation")
    return np.clip(
        observation, -config.observation_clip, config.observation_clip
    ).astype(np.float32)


def normalized_wrench_prediction(wrench):
    wrench = np.asarray(wrench, dtype=np.float64)
    if wrench.shape != (5, 6) or not np.all(np.isfinite(wrench)):
        raise ValueError("wrench prediction must be finite with shape (5, 6)")
    return (wrench * WRENCH_SCALE).reshape(-1).astype(np.float32)


def compose_policy_input(config, encoder, proprio, wrench, command):
    result = np.concatenate((encoder, proprio, normalized_wrench_prediction(wrench), command)).astype(np.float32)
    if result.shape != (config.policy_input_size,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid 78D SFYG policy input")
    return result


def compose_joint_targets(config, actions, solution):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (10,) or not np.all(np.isfinite(actions)):
        raise ValueError("SFYG locomotion action must be finite and 10D")
    targets = config.default_q.copy()
    targets[:10] += config.action_scale * actions
    targets[10:16] = solution.arm_position
    return targets
