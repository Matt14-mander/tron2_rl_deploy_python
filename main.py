import argparse
import os
import sys
import controllers as controllers
# 注意：limxsdk 延迟到 SF/WF 分支内 import——顶层 import 会抢先把 pip 旧版载入
# sys.modules，使 DA_SF --sdk 模式无法优先解析到新版（LIMXSDK_PYTHON_DIR/子仓）。

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TRON2 policy controller entry")
    parser.add_argument("robot_ip", nargs="?", default="127.0.0.1",
                        help="robot ip (default 127.0.0.1)")
    comm_group = parser.add_mutually_exclusive_group()
    comm_group.add_argument("--mros", dest="comm", action="store_const", const="mros",
                            help="DA_SF: 运控通道走 mrospy 直连（默认）")
    comm_group.add_argument("--sdk", dest="comm", action="store_const", const="sdk",
                            help="DA_SF: 运控通道走 limxsdk Centaur controller 端")
    parser.set_defaults(comm="mros")
    args = parser.parse_args()

    # Get the robot type from the environment variable
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    model_dir = f'{os.path.dirname(os.path.abspath(__file__))}/controllers/model'

    if robot_type not in ("SF_TRON2A", "WF_TRON2A", "DA_SF_TRON2A"):
        print(f"\033[31mError: unsupported ROBOT_TYPE='{robot_type}', expected SF_TRON2A, WF_TRON2A, or DA_SF_TRON2A\033[0m")
        sys.exit(1)

    # DA-SF: one Python node; motion channels selectable (--mros default | --sdk),
    # /joystick stays on MROS either way. --sdk uses ROBOT_IP env or robot_ip arg.
    if robot_type == "DA_SF_TRON2A":
        if args.comm == "sdk":
            os.environ.setdefault("ROBOT_IP", args.robot_ip)
        controller = controllers.DASFController(model_dir, robot_type, False, comm=args.comm)
        controller.run()
        sys.exit(0)

    # Create a Robot instance of the specified type
    import limxsdk.robot.Robot as Robot
    import limxsdk.robot.RobotType as RobotType
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
    elif robot_type == "WF_TRON2A":
        controller = controllers.WheelfootController(model_dir, robot, robot_type, start_controller, use_pygame_joystick=use_pygame_joystick)
        controller.run()
