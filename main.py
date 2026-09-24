import argparse
import os
import sys

import controllers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TRON2 policy controller entry")
    parser.add_argument("robot_ip", nargs="?", default="127.0.0.1",
                        help="robot ip (default 127.0.0.1)")
    parser.add_argument("--sdk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ocs2-trajectory", default=None,
                        help="SFYG: 52-column OCS2 trajectory CSV")
    parser.add_argument("--ocs2-start-delay", type=float, default=0.0,
                        help="SFYG: seconds to stabilize before OCS2 arm motion")
    parser.add_argument("--ocs2-terminal-command", nargs=3, type=float, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="SFYG: walking command after OCS2 trajectory ends")
    parser.add_argument("--ocs2-hold-arm", action="store_true",
                        help="SFYG diagnostic: replay base command and wrench, hold arm")
    parser.add_argument("--ocs2-zero-wrench", action="store_true",
                        help="SFYG diagnostic: replay base command and arm, zero policy wrench")
    parser.add_argument("--arm-test-joint", choices=tuple(f"arm{i}" for i in range(1, 7)),
                        help="SFYG sim2sim: move one arm joint out and back")
    parser.add_argument("--arm-test-delta", type=float,
                        help="SFYG sim2sim: signed arm joint displacement in radians (max 0.15)")
    parser.add_argument("--arm-test-offsets", nargs=6, type=float,
                        metavar=("ARM1", "ARM2", "ARM3", "ARM4", "ARM5", "ARM6"),
                        help="SFYG sim2sim: six joint offsets; exactly two nonzero, bounded")
    parser.add_argument("--arm-test-start-delay", type=float, default=3.0,
                        help="SFYG sim2sim: initial walking warmup in seconds")
    parser.add_argument("--arm-test-move-duration", type=float, default=2.0,
                        help="SFYG sim2sim: seconds for each smooth arm movement")
    parser.add_argument("--arm-test-hold-duration", type=float, default=1.0,
                        help="SFYG sim2sim: seconds to hold the displaced arm target")
    parser.add_argument("--start-controller", action="store_true",
                        help="start policy immediately")
    parser.add_argument("--base-command", nargs=3, type=float,
                        metavar=("VX", "VY", "WZ"), default=(0.0, 0.0, 0.0),
                        help="SFYG velocity command without OCS2 (m/s, m/s, rad/s)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="exit after N seconds; 0 runs until interrupted")
    args = parser.parse_args()

    # Get the robot type from the environment variable
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    model_dir = f'{os.path.dirname(os.path.abspath(__file__))}/controllers/model'

    if robot_type not in ("SF_TRON2A", "SFYG_TRON2A", "WF_TRON2A", "DASF_TRON2A"):
        print(f"\033[31mError: unsupported ROBOT_TYPE='{robot_type}'\033[0m")
        sys.exit(1)
    single_arm_test = args.arm_test_joint is not None or args.arm_test_delta is not None
    arm_test_requested = single_arm_test or args.arm_test_offsets is not None
    if single_arm_test and (args.arm_test_joint is None or args.arm_test_delta is None):
        parser.error("arm test requires both --arm-test-joint and --arm-test-delta")
    if single_arm_test and args.arm_test_offsets is not None:
        parser.error("choose either --arm-test-offsets or --arm-test-joint/--arm-test-delta")
    if arm_test_requested and (robot_type != "SFYG_TRON2A" or args.robot_ip != "127.0.0.1"):
        parser.error("arm test is SFYG_TRON2A sim2sim-only (robot_ip must be 127.0.0.1)")
    if arm_test_requested and (args.ocs2_trajectory is not None or not args.start_controller):
        parser.error("arm test requires --start-controller and cannot replay OCS2")

    # DA-SF uses the Centaur SDK for motion, IMU, and joystick channels.
    if robot_type == "DASF_TRON2A":
        os.environ.setdefault("ROBOT_IP", args.robot_ip)
        controller = controllers.DASFController(model_dir, robot_type, False)
        controller.run()
        sys.exit(0)

    # Create a Robot instance of the specified type
    from limxsdk.robot import Robot, RobotType
    robot = Robot(RobotType.Tron2)

    # Initialize the robot with the provided IP address
    if not robot.init(args.robot_ip):
        sys.exit()

    use_pygame_joystick = True

    # Determine if the simulation is running
    start_controller = False

    # Create and run the controller
    if robot_type == "SF_TRON2A":
        controller = controllers.SolefootController(model_dir, robot, robot_type, start_controller, use_pygame_joystick=use_pygame_joystick)
        controller.run()
    elif robot_type == "SFYG_TRON2A":
        controller = controllers.SFYGController(
            model_dir,
            robot,
            robot_type,
            start_controller=args.start_controller,
            ocs2_trajectory=args.ocs2_trajectory,
            base_command=args.base_command,
            ocs2_start_delay=args.ocs2_start_delay,
            ocs2_terminal_command=args.ocs2_terminal_command,
            ocs2_hold_arm=args.ocs2_hold_arm,
            ocs2_zero_wrench=args.ocs2_zero_wrench,
            arm_test_joint=args.arm_test_joint,
            arm_test_delta=args.arm_test_delta,
            arm_test_offsets=args.arm_test_offsets,
            arm_test_start_delay=args.arm_test_start_delay,
            arm_test_move_duration=args.arm_test_move_duration,
            arm_test_hold_duration=args.arm_test_hold_duration,
        )
        controller.run(duration=args.duration)
    elif robot_type == "WF_TRON2A":
        controller = controllers.WheelfootController(model_dir, robot, robot_type, start_controller, use_pygame_joystick=use_pygame_joystick)
        controller.run()
