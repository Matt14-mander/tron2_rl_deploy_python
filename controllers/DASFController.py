#!/usr/bin/env python3
"""DA-SF policy controller for the dual-channel MuJoCo simulator."""

import math
import os
import threading
import time

import numpy as np
import yaml

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    import mros
    import mros.controller_msgs.msg.IMUData
    import mros.controller_msgs.msg.JointCmd
    import mros.controller_msgs.msg.JointState
    import mros.sensor_msgs.msg.Joy
except ImportError:
    mros = None


class DASFPolicyConfig:
    """Validated runtime configuration loaded from params.yaml."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Configuration file not found: {self.path}")
        with open(self.path, encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
        if not isinstance(document, dict) or "PointfootCfg" not in document:
            raise ValueError("params.yaml must contain a PointfootCfg mapping")
        config = document["PointfootCfg"]
        self.robot_variant = str(config["robot_variant"])
        self.xml_path = str(config["xml_path"])

        size = config["size"]
        self.num_lower = 10
        self.num_upper = 16
        self.num_joints = self.num_lower + self.num_upper
        self.num_actions = int(size["actions_size"])
        self.obs_size = int(size["policy_obs_size"])
        self.command_size = int(size["commands_obs_size"])
        self.history_length = int(size["obs_history_length"])
        self.encoder_output_size = int(size["encoder_output_size"])
        self.history_size = self.obs_size * self.history_length
        self.policy_input_size = (
            self.encoder_output_size + self.obs_size + self.command_size
        )
        configured_history_size = int(size["encoder_input_size"])
        configured_policy_input_size = int(size["policy_input_size"])
        if configured_history_size != self.history_size:
            raise ValueError(
                f"encoder_input_size is {configured_history_size}, "
                f"expected {self.history_size}"
            )
        if configured_policy_input_size != self.policy_input_size:
            raise ValueError(
                f"policy_input_size is {configured_policy_input_size}, "
                f"expected {self.policy_input_size}"
            )
        expected_obs_size = 3 + 3 + 3 * self.num_actions + 2
        if self.obs_size != expected_obs_size:
            raise ValueError(
                f"policy_obs_size is {self.obs_size}, expected {expected_obs_size}"
            )
        if self.command_size != 3:
            raise ValueError("DA-SF velocity command must be 3D")

        init_state = config["init_state"]
        self.full_joint_names = list(init_state["joint_names"])
        self.policy_joint_names = list(init_state["policy_joint_names"])
        self._validate_joint_names()
        full_indices = {name: index for index, name in enumerate(self.full_joint_names)}
        self.policy_to_full = np.array(
            [full_indices[name] for name in self.policy_joint_names], dtype=np.int64
        )

        self.default_q = self._ordered_array(
            init_state["default_joint_angle"],
            self.full_joint_names,
            "init_state.default_joint_angle",
        )
        control = config["control"]
        control_modes = list(control["joint_control_modes"])
        if len(control_modes) != self.num_joints or any(
            mode != "position" for mode in control_modes
        ):
            raise ValueError("DA-SF requires 26 position joint_control_modes")
        self.kp = self._ordered_array(
            control["stiffness"], self.full_joint_names, "control.stiffness"
        )
        self.kd = self._ordered_array(
            control["damping"], self.full_joint_names, "control.damping"
        )
        self.action_scales = self._ordered_array(
            control["action_scale_pos"],
            self.policy_joint_names,
            "control.action_scale_pos",
        )

        observation_scales = config["normalization"]["obs_scales"]
        self.ang_vel_scale = float(observation_scales["ang_vel"])
        self.dof_pos_scale = float(observation_scales["dof_pos"])
        self.dof_vel_scale = float(observation_scales["dof_vel"])
        gait = config["gait"]
        self.phase_period = float(gait["period"])
        self.phase_command_threshold = float(gait["command_threshold"])
        if self.phase_period <= 0.0 or self.phase_command_threshold < 0.0:
            raise ValueError("phase period must be positive and threshold non-negative")

        runtime = config["runtime"]
        loop_frequency = float(config["loop_frequency"])
        decimation = int(control["decimation"])
        if loop_frequency <= 0.0 or decimation <= 0:
            raise ValueError("loop_frequency and control.decimation must be positive")
        self.policy_frequency = loop_frequency / decimation
        self.stand_control_frequency = loop_frequency
        self.stand_duration = float(config["stand_mode"]["stand_duration"])
        self.state_timeout = float(runtime["state_timeout"])
        self.state_wait_timeout = float(runtime["state_wait_timeout"])
        runtime_values = (
            self.policy_frequency,
            self.stand_control_frequency,
            self.stand_duration,
            self.state_timeout,
            self.state_wait_timeout,
        )
        if any(value <= 0.0 for value in runtime_values):
            raise ValueError("All runtime frequencies, durations, and timeouts must be positive")

        clip_scales = config["normalization"]["clip_scales"]
        self.observation_clip = abs(float(clip_scales["clip_observations"]))
        self.action_clip = abs(float(clip_scales["clip_actions"]))
        commands = config["commands"]
        command_keys = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        self.default_command = np.asarray(
            [commands["default"][key] for key in command_keys], dtype=np.float64
        )
        self.max_command = np.asarray(
            [commands["max"][key] for key in command_keys], dtype=np.float64
        )
        if self.default_command.shape != (self.command_size,):
            raise ValueError("commands.default has the wrong dimension")
        if self.max_command.shape != (self.command_size,) or np.any(
            self.max_command <= 0.0
        ):
            raise ValueError("commands.max_abs must contain three positive values")

        self.encoder_file = "encoder.onnx"
        self.policy_file = "policy.onnx"
        communication = config["communication"]
        self.lower_cmd_topic = str(communication["lower_cmd_topic"])
        self.lower_state_topic = str(communication["lower_state_topic"])
        self.upper_cmd_topic = str(communication["upper_cmd_topic"])
        self.upper_state_topic = str(communication["upper_state_topic"])
        self.imu_topic = str(communication["imu_topic"])
        self.joystick_topic = str(communication["joystick_topic"])

    def _validate_joint_names(self):
        if len(self.full_joint_names) != self.num_joints:
            raise ValueError("init_state.joint_names length does not match total joints")
        if len(self.policy_joint_names) != self.num_actions:
            raise ValueError(
                "init_state.policy_joint_names length does not match actions_size"
            )
        if len(set(self.full_joint_names)) != len(self.full_joint_names):
            raise ValueError("init_state.joint_names contains duplicate names")
        if len(set(self.policy_joint_names)) != len(self.policy_joint_names):
            raise ValueError("init_state.policy_joint_names contains duplicate names")
        unknown = set(self.policy_joint_names) - set(self.full_joint_names)
        if unknown:
            raise ValueError(
                f"Policy joints missing from joint_names: {sorted(unknown)}"
            )

    @staticmethod
    def _ordered_array(values, names, field_name):
        missing = [name for name in names if name not in values]
        extra = sorted(set(values) - set(names))
        if missing or extra:
            raise ValueError(
                f"{field_name} names mismatch: missing={missing}, extra={extra}"
            )
        result = np.asarray([values[name] for name in names], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{field_name} contains non-finite values")
        return result


def projected_gravity_from_wxyz(quaternion):
    """Rotate world gravity [0, 0, -1] into the body frame."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected a 4D quaternion, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("Received an invalid IMU quaternion")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            2.0 * (x * x + y * y) - 1.0,
        ],
        dtype=np.float64,
    )


def build_policy_observation(
    config,
    full_q,
    full_dq,
    gyro,
    quaternion,
    last_action,
    command,
    policy_elapsed,
):
    """Build the exact 68D policy observation used by DA-SF training."""
    full_q = np.asarray(full_q, dtype=np.float64)
    full_dq = np.asarray(full_dq, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    last_action = np.asarray(last_action, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if full_q.shape != (config.num_joints,) or full_dq.shape != (
        config.num_joints,
    ):
        raise ValueError("DA-SF joint state must contain 26 positions and velocities")
    if gyro.shape != (3,) or command.shape != (config.command_size,):
        raise ValueError("Gyro and velocity command must both be 3D")
    if last_action.shape != (config.num_actions,):
        raise ValueError("Previous DA-SF action must be 20D")

    if np.linalg.norm(command) < config.phase_command_threshold:
        phase = np.zeros(2, dtype=np.float64)
    else:
        phase_angle = 2.0 * math.pi * (
            (policy_elapsed % config.phase_period) / config.phase_period
        )
        phase = np.array([math.sin(phase_angle), math.cos(phase_angle)])

    policy_q = full_q[config.policy_to_full]
    policy_dq = full_dq[config.policy_to_full]
    default_policy_q = config.default_q[config.policy_to_full]
    observation = np.concatenate(
        (
            gyro * config.ang_vel_scale,
            projected_gravity_from_wxyz(quaternion),
            (policy_q - default_policy_q) * config.dof_pos_scale,
            policy_dq * config.dof_vel_scale,
            last_action,
            phase,
        )
    )
    if observation.shape != (config.obs_size,) or not np.all(
        np.isfinite(observation)
    ):
        raise ValueError("Policy observation is invalid")
    return observation.astype(np.float32)


def actions_to_full_targets(config, actions):
    """Map the 20D policy output to 26D position targets."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (config.num_actions,) or not np.all(np.isfinite(actions)):
        raise ValueError("Policy action must be a finite 20D vector")
    targets = config.default_q.copy()
    targets[config.policy_to_full] += config.action_scales * actions
    return targets


class DASFController:
    """DA-SF ONNX policy controller using the simulator's two MROS channels."""

    JOY_BTNS = {"A": 0, "L1": 4, "R1": 5, "X": 2, "Y": 3}
    JOY_AXES = {"left_vertical": 1, "left_horizon": 0, "right_horizon": 2}
    PYGAME_AXES = {"left_vertical": 1, "left_horizon": 0, "right_horizon": 3}

    def __init__(
        self,
        model_dir,
        robot_type,
        start_controller=False,
        command=None,
        stand_duration=None,
        state_timeout=None,
        action_clip=None,
        use_pygame_joystick=True,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required: pip install onnxruntime")
        if mros is None:
            raise RuntimeError("mrospy is required to communicate with the simulator")

        self.model_dir = os.path.abspath(os.path.join(model_dir, robot_type))
        config = DASFPolicyConfig(os.path.join(self.model_dir, "params.yaml"))
        if config.robot_variant != robot_type:
            raise ValueError(
                f"params.yaml robot_variant is '{config.robot_variant}', "
                f"expected '{robot_type}'"
            )
        self.config = config
        if command is None:
            command = config.default_command
        self.command = np.asarray(command, dtype=np.float64)
        if self.command.shape != (config.command_size,):
            raise ValueError("Velocity command must be [vx, vy, wz]")
        if np.any(np.abs(self.command) > config.max_command):
            raise ValueError(
                f"Velocity command {self.command.tolist()} exceeds max_abs "
                f"{config.max_command.tolist()}"
            )
        self.stand_duration = float(
            config.stand_duration if stand_duration is None else stand_duration
        )
        self.state_timeout = float(
            config.state_timeout if state_timeout is None else state_timeout
        )
        self.action_clip = abs(
            float(config.action_clip if action_clip is None else action_clip)
        )
        if self.stand_duration <= 0.0 or self.state_timeout <= 0.0:
            raise ValueError("stand_duration and state_timeout must be positive")

        self.encoder_session = self._load_session(config.encoder_file)
        self.policy_session = self._load_session(config.policy_file)
        self._validate_model_contract()
        mros.init("da_sf_policy_controller")

        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._lower_q = np.zeros(config.num_lower)
        self._lower_dq = np.zeros(config.num_lower)
        self._upper_q = np.zeros(config.num_upper)
        self._upper_dq = np.zeros(config.num_upper)
        self._quaternion = np.zeros(4)
        self._gyro = np.zeros(3)
        self._lower_stamp = None
        self._upper_stamp = None
        self._imu_stamp = None
        self._lower_count = 0
        self._upper_count = 0
        self._imu_count = 0
        self._lower_invalid = None
        self._upper_invalid = None
        self._imu_invalid = None
        self.start_controller = bool(start_controller)
        self.mode = "WALK" if self.start_controller else "STAND"
        self._history_reset_requested = True
        self._walk_start_time = time.monotonic()
        self._last_r1 = 0
        self._pygame_last_r1 = 0
        self._use_pygame_joystick = bool(use_pygame_joystick)
        self._pygame_enabled = False
        self._pygame_joystick = None
        self.joy_msg_count = 0
        self.last_joy_time = 0.0

        self.pub_lower = mros.advertise(
            config.lower_cmd_topic, mros.controller_msgs.msg.JointCmd
        )
        self.pub_upper = mros.advertise(
            config.upper_cmd_topic, mros.controller_msgs.msg.JointCmd
        )
        self.sub_lower = mros.subscribe(
            config.lower_state_topic,
            mros.controller_msgs.msg.JointState,
            self._lower_state_callback,
        )
        self.sub_upper = mros.subscribe(
            config.upper_state_topic,
            mros.controller_msgs.msg.JointState,
            self._upper_state_callback,
        )
        self.sub_imu = mros.subscribe(
            config.imu_topic, mros.controller_msgs.msg.IMUData, self._imu_callback
        )
        self.sub_joystick = mros.subscribe(
            config.joystick_topic,
            mros.sensor_msgs.msg.Joy,
            self.sensor_joy_callback,
        )

        self.last_action = np.zeros(config.num_actions, dtype=np.float64)
        self.history = np.zeros(config.history_size, dtype=np.float32)
        self._setup_pygame_joystick()
        print(f"Loaded DA-SF parameters from {config.path}")
        print(f"Loaded DA-SF models from {self.model_dir}")
        print("MROS topics ready: lower + did_upbody, base IMU")
        joystick_source = (
            "direct pygame/F710" if self._pygame_enabled else "MROS /joystick"
        )
        print(f"Joystick source: {joystick_source}")
        print("Joystick: L1 + Y start, L1 + X stop, R1 clear command/history")

    def _load_session(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
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
        if not shape or not isinstance(shape[-1], int):
            return None
        return shape[-1]

    def _validate_model_contract(self):
        encoder_inputs = self.encoder_session.get_inputs()
        encoder_outputs = self.encoder_session.get_outputs()
        policy_inputs = self.policy_session.get_inputs()
        policy_outputs = self.policy_session.get_outputs()
        if len(encoder_inputs) != 1 or len(encoder_outputs) != 1:
            raise ValueError("Encoder ONNX must have exactly one input and one output")
        if len(policy_inputs) != 1 or len(policy_outputs) != 1:
            raise ValueError("Policy ONNX must have exactly one input and one output")

        contracts = (
            ("encoder input", encoder_inputs[0], self.config.history_size),
            (
                "encoder output",
                encoder_outputs[0],
                self.config.encoder_output_size,
            ),
            ("policy input", policy_inputs[0], self.config.policy_input_size),
            ("policy output", policy_outputs[0], self.config.num_actions),
        )
        for name, value_info, expected in contracts:
            actual = self._feature_size(value_info)
            if actual is not None and actual != expected:
                raise ValueError(f"{name} is {actual}D, expected {expected}D")
        print(
            f"Model contract: encoder {self.config.history_size}->"
            f"{self.config.encoder_output_size}, policy "
            f"{self.config.policy_input_size}->{self.config.num_actions} "
            "(policy includes actor normalization)"
        )

    def _lower_state_callback(self, message):
        if (
            len(message.q) < self.config.num_lower
            or len(message.v) < self.config.num_lower
        ):
            self._lower_invalid = (len(message.q), len(message.v))
            return
        with self._lock:
            self._lower_q[:] = message.q[: self.config.num_lower]
            self._lower_dq[:] = message.v[: self.config.num_lower]
            self._lower_stamp = time.monotonic()
            self._lower_count += 1

    def _upper_state_callback(self, message):
        if (
            len(message.q) < self.config.num_upper
            or len(message.v) < self.config.num_upper
        ):
            self._upper_invalid = (len(message.q), len(message.v))
            return
        with self._lock:
            self._upper_q[:] = message.q[: self.config.num_upper]
            self._upper_dq[:] = message.v[: self.config.num_upper]
            self._upper_stamp = time.monotonic()
            self._upper_count += 1

    def _imu_callback(self, message):
        if len(message.quat) < 4 or len(message.gyro) < 3:
            self._imu_invalid = (len(message.quat), len(message.gyro))
            return
        with self._lock:
            self._quaternion[:] = message.quat[:4]
            self._gyro[:] = message.gyro[:3]
            self._imu_stamp = time.monotonic()
            self._imu_count += 1

    @staticmethod
    def _clip_unit(value):
        return max(-1.0, min(1.0, float(value)))

    def _setup_pygame_joystick(self):
        if not self._use_pygame_joystick:
            return
        if pygame is None:
            print("pygame is unavailable; using MROS /joystick")
            return
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                print("No direct joystick found; using MROS /joystick")
                return
            self._pygame_joystick = pygame.joystick.Joystick(0)
            self._pygame_joystick.init()
            self._pygame_enabled = True
            print(f"Direct joystick ready: {self._pygame_joystick.get_name()}")
        except pygame.error as error:
            self._pygame_joystick = None
            print(f"Direct joystick init failed ({error}); using MROS /joystick")

    @staticmethod
    def _deadzone(value, threshold=0.08):
        return 0.0 if abs(value) < threshold else value

    def _process_joystick(self, buttons, axes, r1_state_attr="_last_r1"):
        l1 = buttons[self.JOY_BTNS["L1"]] if len(buttons) > self.JOY_BTNS["L1"] else 0
        x_btn = buttons[self.JOY_BTNS["X"]] if len(buttons) > self.JOY_BTNS["X"] else 0
        y_btn = buttons[self.JOY_BTNS["Y"]] if len(buttons) > self.JOY_BTNS["Y"] else 0
        r1 = buttons[self.JOY_BTNS["R1"]] if len(buttons) > self.JOY_BTNS["R1"] else 0

        with self._control_lock:
            if not self.start_controller and l1 and y_btn:
                print("L1 + Y: start DA-SF policy")
                self.start_controller = True
                self.mode = "WALK"
                self._history_reset_requested = True
                self._walk_start_time = time.monotonic()

            if self.start_controller and l1 and x_btn:
                print("L1 + X: stop DA-SF policy")
                self.start_controller = False
                self.mode = "IDLE"
                self.command.fill(0.0)
                self._history_reset_requested = True

            if r1 and not getattr(self, r1_state_attr):
                self.command.fill(0.0)
                self._history_reset_requested = True
            setattr(self, r1_state_attr, int(r1))

            if self.mode != "WALK":
                return

            normalized = np.array(
                [
                    self._clip_unit(axes[0] if len(axes) > 0 else 0.0),
                    self._clip_unit(axes[1] if len(axes) > 1 else 0.0),
                    self._clip_unit(axes[2] if len(axes) > 2 else 0.0),
                ],
                dtype=np.float64,
            )
            normalized[np.abs(normalized) < 0.08] = 0.0
            self.command[:] = normalized * self.config.max_command

    def _poll_pygame_joystick(self):
        if not self._pygame_enabled or self._pygame_joystick is None:
            return
        pygame.event.pump()
        joystick = self._pygame_joystick
        num_axes = joystick.get_numaxes()
        num_buttons = joystick.get_numbuttons()
        buttons = [
            joystick.get_button(index) if index < num_buttons else 0
            for index in range(6)
        ]
        left_horizontal = (
            joystick.get_axis(self.PYGAME_AXES["left_horizon"])
            if num_axes > self.PYGAME_AXES["left_horizon"]
            else 0.0
        )
        left_vertical = (
            joystick.get_axis(self.PYGAME_AXES["left_vertical"])
            if num_axes > self.PYGAME_AXES["left_vertical"]
            else 0.0
        )
        right_horizontal = (
            joystick.get_axis(self.PYGAME_AXES["right_horizon"])
            if num_axes > self.PYGAME_AXES["right_horizon"]
            else 0.0
        )
        axes = [
            -self._deadzone(left_vertical),
            -self._deadzone(left_horizontal),
            -self._deadzone(right_horizontal),
        ]
        self._process_joystick(buttons, axes, "_pygame_last_r1")
        self.joy_msg_count += 1
        self.last_joy_time = time.monotonic()

    def sensor_joy_callback(self, sensor_joy):
        if getattr(self, "_pygame_enabled", False):
            return
        self.joy_msg_count += 1
        self.last_joy_time = time.monotonic()
        axes = [
            sensor_joy.axes[self.JOY_AXES["left_vertical"]]
            if len(sensor_joy.axes) > self.JOY_AXES["left_vertical"]
            else 0.0,
            sensor_joy.axes[self.JOY_AXES["left_horizon"]]
            if len(sensor_joy.axes) > self.JOY_AXES["left_horizon"]
            else 0.0,
            sensor_joy.axes[self.JOY_AXES["right_horizon"]]
            if len(sensor_joy.axes) > self.JOY_AXES["right_horizon"]
            else 0.0,
        ]
        self._process_joystick(sensor_joy.buttons, axes, "_last_r1")

    def joystick_alive(self, timeout_sec=2.0):
        return self.joy_msg_count > 0 and (
            time.monotonic() - self.last_joy_time
        ) <= float(timeout_sec)

    def _consume_control_state(self):
        with self._control_lock:
            state = (
                self.start_controller,
                self.mode,
                self.command.copy(),
                self._history_reset_requested,
                self._walk_start_time,
            )
            self._history_reset_requested = False
        return state

    def _snapshot(self, require_imu=True):
        now = time.monotonic()
        with self._lock:
            stamps = [self._lower_stamp, self._upper_stamp]
            if require_imu:
                stamps.append(self._imu_stamp)
            if any(stamp is None for stamp in stamps):
                raise RuntimeError("DA-SF state is not ready")
            if any(now - stamp > self.state_timeout for stamp in stamps):
                raise RuntimeError("DA-SF state stream timed out")
            q = np.concatenate((self._lower_q, self._upper_q))
            dq = np.concatenate((self._lower_dq, self._upper_dq))
            quaternion = self._quaternion.copy()
            gyro = self._gyro.copy()
        return q, dq, quaternion, gyro

    @staticmethod
    def _make_joint_command(q, kp, kd):
        message = mros.controller_msgs.msg.JointCmd()
        num_joints = len(q)
        message.q = list(q)
        message.v = [0.0] * num_joints
        message.tau = [0.0] * num_joints
        message.kp = list(kp)
        message.kd = list(kd)
        message.mode = [10] * num_joints
        message.na = num_joints
        return message

    def _publish_targets(self, targets, kp=None, kd=None):
        targets = np.asarray(targets, dtype=np.float64)
        if kp is None:
            kp = self.config.kp
        if kd is None:
            kd = self.config.kd
        lower = self.config.num_lower
        self.pub_lower.publish(
            self._make_joint_command(
                targets[:lower], kp[:lower], kd[:lower]
            )
        )
        self.pub_upper.publish(
            self._make_joint_command(
                targets[lower:], kp[lower:], kd[lower:]
            )
        )

    def _wait_for_state(self, timeout=None):
        print("Waiting for lower state, upper state, and base IMU...")
        if timeout is None:
            timeout = self.config.state_wait_timeout
        deadline = time.monotonic() + timeout
        next_report = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                q, _, _, _ = self._snapshot(require_imu=False)
                self._publish_targets(q)
            except RuntimeError:
                pass
            try:
                snapshot = self._snapshot(require_imu=True)
                print("All DA-SF state streams are ready")
                return snapshot
            except RuntimeError:
                if time.monotonic() >= next_report:
                    print(self._state_diagnostics())
                    next_report += 1.0
                time.sleep(0.005)
        raise RuntimeError(
            "Timed out waiting for simulator DA-SF state. "
            + self._state_diagnostics()
        )

    @staticmethod
    def _subscriber_status(subscriber):
        try:
            return "yes" if subscriber.negotiated() else "no"
        except Exception:
            return "unknown"

    def _state_diagnostics(self):
        now = time.monotonic()

        def age(stamp):
            return "never" if stamp is None else f"{now - stamp:.3f}s"

        with self._lock:
            lower = (self._lower_count, age(self._lower_stamp), self._lower_invalid)
            upper = (self._upper_count, age(self._upper_stamp), self._upper_invalid)
            imu = (self._imu_count, age(self._imu_stamp), self._imu_invalid)
        return (
            "streams: "
            f"lower(negotiated={self._subscriber_status(self.sub_lower)}, "
            f"count={lower[0]}, age={lower[1]}, invalid={lower[2]}); "
            f"upper(negotiated={self._subscriber_status(self.sub_upper)}, "
            f"count={upper[0]}, age={upper[1]}, invalid={upper[2]}); "
            f"imu(negotiated={self._subscriber_status(self.sub_imu)}, "
            f"count={imu[0]}, age={imu[1]}, invalid={imu[2]}); "
            f"joystick(negotiated={self._subscriber_status(self.sub_joystick)}, "
            f"count={self.joy_msg_count})"
        )

    def _move_to_default_pose(self, start_q):
        print(f"Moving to training default pose over {self.stand_duration:.1f}s")
        start_time = time.monotonic()
        next_tick = start_time
        while True:
            self._snapshot(require_imu=True)
            elapsed = time.monotonic() - start_time
            ratio = min(elapsed / self.stand_duration, 1.0)
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            targets = (
                start_q * (1.0 - smooth_ratio)
                + self.config.default_q * smooth_ratio
            )
            self._publish_targets(targets)
            if ratio >= 1.0:
                break
            next_tick += 1.0 / self.config.stand_control_frequency
            time.sleep(max(0.0, next_tick - time.monotonic()))
        print("Training default pose reached")

    @staticmethod
    def _onnx_input(session, vector):
        vector = np.asarray(vector, dtype=np.float32)
        input_info = session.get_inputs()[0]
        if len(input_info.shape) == 2:
            vector = vector.reshape(1, -1)
        return {input_info.name: vector}

    def _infer(self, observation, command):
        obs_size = self.config.obs_size
        self.history[:-obs_size] = self.history[obs_size:]
        self.history[-obs_size:] = observation
        encoder_output = self.encoder_session.run(
            None, self._onnx_input(self.encoder_session, self.history)
        )[0]
        encoder_output = np.asarray(encoder_output, dtype=np.float32).reshape(-1)
        policy_input = np.concatenate(
            (encoder_output, observation, command.astype(np.float32))
        )
        actions = self.policy_session.run(
            None, self._onnx_input(self.policy_session, policy_input)
        )[0]
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        if actions.shape != (self.config.num_actions,) or not np.all(
            np.isfinite(actions)
        ):
            raise RuntimeError("Policy returned an invalid action")
        return np.clip(actions, -self.action_clip, self.action_clip)

    def _publish_damping_stop(self):
        try:
            q, _, _, _ = self._snapshot(require_imu=False)
        except RuntimeError:
            return
        self._publish_targets(
            q, kp=np.zeros(self.config.num_joints), kd=self.config.kd
        )
        print("Published damping stop command")

    def run(self, duration=0.0):
        try:
            initial_q, _, _, _ = self._wait_for_state()
            self._move_to_default_pose(initial_q)
            print(
                f"DA-SF ready at {self.config.policy_frequency:g} Hz; "
                "press L1 + Y to start"
            )
            start_time = time.monotonic()
            next_tick = start_time
            last_status = start_time
            while duration <= 0.0 or time.monotonic() - start_time < duration:
                self._poll_pygame_joystick()
                q, dq, quaternion, gyro = self._snapshot(require_imu=True)
                (
                    policy_enabled,
                    mode,
                    command,
                    reset_history,
                    walk_start_time,
                ) = self._consume_control_state()
                if not policy_enabled:
                    if mode == "STAND":
                        self._publish_targets(self.config.default_q)
                    else:
                        self._publish_targets(
                            q,
                            kp=np.zeros(self.config.num_joints),
                            kd=self.config.kd,
                        )
                    policy_period = 1.0 / self.config.policy_frequency
                    next_tick += policy_period
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                    continue

                policy_elapsed = time.monotonic() - walk_start_time
                if reset_history:
                    self.last_action.fill(0.0)
                observation = build_policy_observation(
                    self.config,
                    q,
                    dq,
                    gyro,
                    quaternion,
                    self.last_action,
                    command,
                    policy_elapsed,
                )
                if reset_history:
                    self.history[:] = np.tile(
                        observation, self.config.history_length
                    )
                self.last_action = self._infer(observation, command)
                self._publish_targets(
                    actions_to_full_targets(self.config, self.last_action)
                )

                if time.monotonic() - last_status >= 1.0:
                    print(
                        f"t={policy_elapsed:7.2f}s  "
                        f"cmd=[{command[0]:+.2f}, {command[1]:+.2f}, "
                        f"{command[2]:+.2f}]  "
                        f"max|action|={np.max(np.abs(self.last_action)):.3f}  "
                        f"joy={'ok' if self.joystick_alive() else 'offline'}"
                    )
                    last_status = time.monotonic()

                policy_period = 1.0 / self.config.policy_frequency
                next_tick += policy_period
                if next_tick < time.monotonic() - policy_period:
                    next_tick = time.monotonic()
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            print("Controller interrupted")
        except RuntimeError as error:
            print(f"DA-SF controller stopped: {error}")
        finally:
            self._publish_damping_stop()
            try:
                mros.shutdown()
            except Exception as error:
                print(f"MROS shutdown warning: {error}")

