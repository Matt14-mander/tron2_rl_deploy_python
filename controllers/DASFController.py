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
        # MROS 下发携带的关节名（线序真名 = config 短名 + "_Joint"，与仿真/真机
        # state.names 一致）。CentaurSim 按 cmd.names 与 state.names 匹配校验，
        # 无名 cmd 会被 native 拦截（"RobotCmd does not match the corresponding
        # RobotState"）。硬件坐标变换为逐元素方向/零位映射、不重排序，故名字
        # 顺序与下发数组顺序始终一致。
        self.wire_joint_names = [f"{name}_Joint" for name in self.full_joint_names]
        full_indices = {name: index for index, name in enumerate(self.full_joint_names)}
        self.hardware_joint_zero = np.zeros(self.num_joints, dtype=np.float64)
        self.hardware_joint_direction = np.ones(self.num_joints, dtype=np.float64)
        self.hardware_joint_zero[full_indices["proximal_yaw_L"]] = -math.pi
        self.hardware_joint_zero[full_indices["proximal_yaw_R"]] = math.pi
        for name in ("knee_L", "ankle_pitch_L", "knee_R", "ankle_pitch_R"):
            self.hardware_joint_direction[full_indices[name]] = -1.0
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


def hardware_to_policy_state(config, hardware_q, hardware_dq):
    """Convert real-robot joint state to the coordinates used in training."""
    hardware_q = np.asarray(hardware_q, dtype=np.float64)
    hardware_dq = np.asarray(hardware_dq, dtype=np.float64)
    expected_shape = (config.num_joints,)
    if hardware_q.shape != expected_shape or hardware_dq.shape != expected_shape:
        raise ValueError("Hardware joint state must contain 26 positions and velocities")
    policy_q = config.hardware_joint_direction * (
        hardware_q - config.hardware_joint_zero
    )
    policy_dq = config.hardware_joint_direction * hardware_dq
    return policy_q, policy_dq


def policy_targets_to_hardware(config, policy_targets):
    """Convert policy-coordinate position targets to real-robot coordinates."""
    policy_targets = np.asarray(policy_targets, dtype=np.float64)
    if policy_targets.shape != (config.num_joints,):
        raise ValueError("Policy targets must contain 26 joint positions")
    return (
        config.hardware_joint_direction * policy_targets
        + config.hardware_joint_zero
    )


def hardware_joint_transform_enabled(environment=None):
    """Enable the adapter for MROS hardware targets unless explicitly overridden."""
    environment = os.environ if environment is None else environment
    override = environment.get("DA_SF_HARDWARE_JOINT_TRANSFORM")
    if override is not None:
        normalized = override.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            "DA_SF_HARDWARE_JOINT_TRANSFORM must be 0/1 or false/true"
        )
    return any(
        environment.get(name)
        for name in ("MROS_AGENT_IP", "MROS_AGENT_URI", "MROS_IP_LIST")
    )


class DASFController:
    """DA-SF ONNX policy controller using the simulator's two MROS channels."""

    # 运控通道模式类级默认（"mros" | "sdk"）：契约测试等场景绕过 __init__ 构造
    # 实例时也能取到默认值。
    comm_mode = "mros"

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
        comm="mros",
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required: pip install onnxruntime")
        if mros is None:
            raise RuntimeError("mrospy is required to communicate with the simulator")
        if comm not in ("mros", "sdk"):
            raise ValueError(f"comm must be 'mros' or 'sdk', got {comm!r}")
        self.comm_mode = comm

        self.model_dir = os.path.abspath(os.path.join(model_dir, robot_type))
        config = DASFPolicyConfig(os.path.join(self.model_dir, "params.yaml"))
        if config.robot_variant != robot_type:
            raise ValueError(
                f"params.yaml robot_variant is '{config.robot_variant}', "
                f"expected '{robot_type}'"
            )
        self.config = config
        self.use_hardware_joint_transform = hardware_joint_transform_enabled()
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

        # 运控通道（cmd/state/IMU）双模式：
        #   mros —— mrospy 直连话题（默认，历史路径）；
        #   sdk  —— limxsdk Centaur controller 端（publishLower/UpperBodyRobotCmd
        #           + subscribeLower/UpperBodyRobotState + subscribeLowerBodyImuData，
        #           与真机部署同款 API；底层同一条 MROS 总线、同话题同消息）。
        # 手柄 /joystick 两种模式下均保持 mros 订阅（非运控通道）。
        self.pub_lower = None
        self.pub_upper = None
        self.sdk_robot = None
        if self.comm_mode == "sdk":
            import limxsdk.datatypes as limx_datatypes
            from limxsdk.robot.Robot import Robot as LimxRobot
            from limxsdk.robot.RobotType import RobotType as LimxRobotType

            if not hasattr(LimxRobotType, "Centaur"):
                raise RuntimeError(
                    "installed limxsdk has no RobotType.Centaur (too old); "
                    "pip install --force-reinstall limxsdk-lowlevel/python3/<arch>/limxsdk-*.whl"
                )
            self._limx_datatypes = limx_datatypes
            self.sdk_robot = LimxRobot(LimxRobotType.Centaur, False)
            robot_ip = os.environ.get("ROBOT_IP", "127.0.0.1")
            if not self.sdk_robot.init(robot_ip):
                raise RuntimeError(f"limxsdk Centaur init failed (robot_ip={robot_ip})")
            self.sdk_robot.subscribeLowerBodyRobotState(self._sdk_lower_state_callback)
            self.sdk_robot.subscribeUpperBodyRobotState(self._sdk_upper_state_callback)
            # datatypes.ImuData 与 mros IMUData 的 quat/gyro 字段同名同长，直接复用回调
            self.sdk_robot.subscribeLowerBodyImuData(self._imu_callback)
        else:
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
        coordinate_source = (
            "real hardware" if self.use_hardware_joint_transform else "simulation"
        )
        print(f"Joint coordinates: {coordinate_source}")
        if self.comm_mode == "sdk":
            print("Comm: limxsdk Centaur (LowerBody/UpperBody cmd+state, base IMU); /joystick via MROS")
        else:
            print("Comm: MROS topics ready: lower + did_upbody, base IMU")
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

    # ---- SDK(Centaur) 模式回调：datatypes.RobotState 的速度字段为 dq（mros 为 v）----
    def _sdk_lower_state_callback(self, state):
        if (
            len(state.q) < self.config.num_lower
            or len(state.dq) < self.config.num_lower
        ):
            self._lower_invalid = (len(state.q), len(state.dq))
            return
        with self._lock:
            self._lower_q[:] = state.q[: self.config.num_lower]
            self._lower_dq[:] = state.dq[: self.config.num_lower]
            self._lower_stamp = time.monotonic()
            self._lower_count += 1

    def _sdk_upper_state_callback(self, state):
        if (
            len(state.q) < self.config.num_upper
            or len(state.dq) < self.config.num_upper
        ):
            self._upper_invalid = (len(state.q), len(state.dq))
            return
        with self._lock:
            self._upper_q[:] = state.q[: self.config.num_upper]
            self._upper_dq[:] = state.dq[: self.config.num_upper]
            self._upper_stamp = time.monotonic()
            self._upper_count += 1

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
        if self.use_hardware_joint_transform:
            q, dq = hardware_to_policy_state(self.config, q, dq)
        return q, dq, quaternion, gyro

    @staticmethod
    def _make_joint_command(q, kp, kd, names):
        message = mros.controller_msgs.msg.JointCmd()
        num_joints = len(q)
        # controller_msgs/JointCmd 的关节名字段为 names（SDK datatypes.RobotCmd
        # 侧对应 motor_names）；须与 q/kp/kd 数组顺序一一对应。
        message.names = list(names)
        message.q = list(q)
        message.v = [0.0] * num_joints
        message.tau = [0.0] * num_joints
        message.kp = list(kp)
        message.kd = list(kd)
        message.mode = [10] * num_joints
        message.na = num_joints
        return message

    def _make_sdk_command(self, q, kp, kd, names):
        """limxsdk datatypes.RobotCmd（SDK 端字段：dq/Kp/Kd/motor_names）。"""
        command = self._limx_datatypes.RobotCmd()
        command.stamp = time.time_ns()
        num_joints = len(q)
        command.mode = [10] * num_joints
        command.q = list(q)
        command.dq = [0.0] * num_joints
        command.tau = [0.0] * num_joints
        command.Kp = list(kp)
        command.Kd = list(kd)
        command.motor_names = list(names)
        command.parallel_solve_required = [False] * num_joints
        return command

    def _publish_targets(self, targets, kp=None, kd=None):
        targets = np.asarray(targets, dtype=np.float64)
        if self.use_hardware_joint_transform:
            targets = policy_targets_to_hardware(self.config, targets)
        if kp is None:
            kp = self.config.kp
        if kd is None:
            kd = self.config.kd
        lower = self.config.num_lower
        names = self.config.wire_joint_names
        if self.comm_mode == "sdk":
            self.sdk_robot.publishLowerBodyRobotCmd(
                self._make_sdk_command(
                    targets[:lower], kp[:lower], kd[:lower], names[:lower]
                )
            )
            self.sdk_robot.publishUpperBodyRobotCmd(
                self._make_sdk_command(
                    targets[lower:], kp[lower:], kd[lower:], names[lower:]
                )
            )
            return
        self.pub_lower.publish(
            self._make_joint_command(
                targets[:lower], kp[:lower], kd[:lower], names[:lower]
            )
        )
        self.pub_upper.publish(
            self._make_joint_command(
                targets[lower:], kp[lower:], kd[lower:], names[lower:]
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

