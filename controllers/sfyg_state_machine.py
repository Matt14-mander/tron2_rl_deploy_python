"""Fail-closed SFYG walking/standing transition contract.

Only simulator ground-truth telemetry can currently establish double support:
the LimX RobotState/ImuData streams do not contain foot contact or base speed.
"""

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class SupportState:
    stamp: float  # local monotonic receive time, not remote simulation time
    base_speed_xy: float
    left_normal_force: float
    right_normal_force: float


class SafeGaitStateMachine:
    """Rate-limit OCS2 commands and guard an opt-in simulator PD stand."""

    def __init__(self, stable_command, *, allow_pd_stand=False):
        self.stable_command = np.asarray(stable_command, dtype=float)
        if (self.stable_command.shape != (3,) or not np.isfinite(self.stable_command).all()
                or np.linalg.norm(self.stable_command[:2]) < 0.02
                or np.any(np.abs(self.stable_command) > [1.0, 0.5, 1.5])):
            raise ValueError("stable walk command must be finite, nonzero and within training limits")
        self.allow_pd_stand = bool(allow_pd_stand)
        self.mode = "WALK"
        self.command = self.stable_command.copy()
        self._last_time = None
        self._support_since = None
        self._transition_start = None
        self._stand_since = None
        self._timeout = False
        self._policy_weight = 1.0
        self._walk_entry_weight = 0.0

    def solver_timeout(self):
        """A missing online solution requests braking, never a zero command."""
        self._timeout = True

    def solver_recovered(self):
        """Call only after a fresh, validated online OCS2 solution arrives."""
        self._timeout = False

    @staticmethod
    def _slew(current, target, dt):
        return current + np.clip(target - current, -np.array([0.25, 0.25, 0.6]) * dt,
                                 np.array([0.25, 0.25, 0.6]) * dt)

    @staticmethod
    def _support_ok(support, now, tilt_rad, gyro):
        return (
            support is not None
            and 0.0 <= now - support.stamp <= 0.15
            and math.isfinite(support.base_speed_xy)
            and support.base_speed_xy <= 0.08
            and support.left_normal_force >= 15.0
            and support.right_normal_force >= 15.0
            and math.isfinite(tilt_rad) and tilt_rad <= 0.12
            and np.isfinite(gyro).all() and np.linalg.norm(gyro[:2]) <= 0.2
        )

    def update(self, requested_command, support, tilt_rad, gyro, now, *,
               resume=False, leg_ready=True, arm_ready=True):
        requested = np.asarray(requested_command, dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            self.solver_timeout()
            requested = self.stable_command
        if np.any(np.abs(requested) > [1.0, 0.5, 1.5]):
            self.solver_timeout()
            requested = self.stable_command
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        stop_requested = self._timeout or np.linalg.norm(requested) < 0.05
        safe = self._support_ok(support, now, tilt_rad, np.asarray(gyro))

        if self.mode == "STAND":
            if resume or not safe or not stop_requested:
                self.mode = "WALK_ENTRY"
                self._transition_start = now
                self._walk_entry_weight = 0.0
                self.command = self.stable_command.copy()
            return self.command.copy(), self.mode, 0.0

        if self.mode == "WALK_ENTRY":
            alpha = min((now - self._transition_start) / 0.8, 1.0)
            weight = self._walk_entry_weight + (1.0 - self._walk_entry_weight) * alpha
            self._policy_weight = weight
            if alpha >= 1.0:
                self.mode = "WALK"
                self._timeout = False
            return self.command.copy(), self.mode, weight

        if stop_requested:
            self.mode = "BRAKE" if self.mode == "WALK" else self.mode
            target = self.stable_command
        else:
            self._timeout = False
            target = requested
            if self.mode == "BRAKE":
                self.mode = "WALK"
                self._support_since = None
        self.command = self._slew(self.command, target, dt)

        if self.mode == "BRAKE" and self.allow_pd_stand and safe and leg_ready and arm_ready and np.allclose(
                self.command, self.stable_command, atol=1e-3):
            if self._support_since is None:
                self._support_since = now
            if now - self._support_since >= 0.6:
                self.mode = "STAND_ENTRY"
                self._transition_start = now
        elif self.mode == "BRAKE":
            self._support_since = None

        if self.mode == "STAND_ENTRY":
            if not safe or not leg_ready or not arm_ready or not stop_requested:
                self.mode = "WALK_ENTRY"
                self._transition_start = now
                self._walk_entry_weight = self._policy_weight
                self._support_since = None
                return self.command.copy(), self.mode, self._policy_weight
            alpha = min((now - self._transition_start) / 0.8, 1.0)
            self._policy_weight = 1.0 - alpha
            if alpha >= 1.0:
                self.mode = "STAND"
                self._stand_since = now
            return self.command.copy(), self.mode, self._policy_weight
        self._policy_weight = 1.0
        return self.command.copy(), self.mode, 1.0


def with_base_command(solution, command):
    """Preserve the arm and wrench while replacing the unsafe base command."""
    return replace(solution, base_command=np.asarray(command, dtype=float).copy())
