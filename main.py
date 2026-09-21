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
    parser.add_argument("--start-controller", action="store_true",
                        help="start policy immediately (SFYG requires OCS2 trajectory)")
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
        )
        controller.run(duration=args.duration)
    elif robot_type == "WF_TRON2A":
        controller = controllers.WheelfootController(model_dir, robot, robot_type, start_controller, use_pygame_joystick=use_pygame_joystick)
        controller.run()
