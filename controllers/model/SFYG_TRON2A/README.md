# SFYG model contract

Place the WholeBody artifacts here:

- `encoder.onnx`: one input `420`, one output `3`;
- `policy.onnx`: one input `78`, one output `10`.

The policy input order is `encoder(3) + proprioception(42) + normalized future
wrench(30) + base velocity command(3)`. Do not copy the ordinary SFYG/SF
`48 -> 10` policy into this directory.

## Guarded gait switching (MuJoCo only)

An offline OCS2 replay now requires a nonzero gait floor, supplied by
`--safe-walk-command VX VY WZ` or by the existing nonzero
`--ocs2-terminal-command`. The controller slews toward that floor when an OCS2
sample requests zero velocity. It does **not** interpret a zero command as a
validated standing gait. An online solver must call
`SFYGController.notify_ocs2_timeout()` on timeout and
`controller.gait_guard.solver_recovered()` only after a fresh validated result;
the current CLI still replays CSV, not a live OCS2 stream.

The `--sim-pd-stand` experiment is disabled by default and is rejected for any
non-loopback robot IP. It requires `--sim-support-port PORT` and the matching
MuJoCo `--sfyg-support-port PORT`. Entry needs at least 0.6 s of fresh telemetry
showing both feet loaded, planar base speed <=0.08 m/s, tilt <=0.12 rad,
low roll/pitch rate, and leg/arm positions and velocities near the stand pose.
The ten leg targets blend over 0.8 s to the configured default pose; measured
mid-swing joint angles are never latched. Missing telemetry prevents entry.
This is an **experimental simulator test**, not a proven real-robot stand mode.

First verify static default-pose PD standing separately. Then find a nonzero
low-speed command that remains stable in MuJoCo, and use that as the gait floor.
The previously tested `-0.29 -0.08 0` is a safe *walking* fallback but is not
evidence that the base speed will meet the stand-entry threshold. If that
threshold is never met, remaining in guarded walking is intentional.
