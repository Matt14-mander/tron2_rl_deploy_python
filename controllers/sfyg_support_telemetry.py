"""Local MuJoCo-only UDP ground-truth support receiver."""

import json
import socket
import time

from .sfyg_state_machine import SupportState


class SimSupportReceiver:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", int(port)))
        self.socket.setblocking(False)
        self.latest = None

    def poll(self):
        while True:
            try:
                packet, _ = self.socket.recvfrom(4096)
            except BlockingIOError:
                return self.latest
            try:
                values = json.loads(packet)
                self.latest = SupportState(
                    stamp=time.monotonic(),
                    base_speed_xy=float(values["base_speed_xy"]),
                    left_normal_force=float(values["left_normal_force"]),
                    right_normal_force=float(values["right_normal_force"]),
                )
            except (ValueError, TypeError, KeyError):
                self.latest = None

    def close(self):
        self.socket.close()
