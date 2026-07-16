# Security Policy

## Scope

`tron2-rl-deploy-python` ships **real-hardware control code** for the
TRON2A humanoid platform. It contains:

- The controller entry point (`main.py`) that opens a network
  connection to a physical robot via the LimX SDK.
- Two Python controllers (`SolefootController`, `WheelfootController`)
  that load ONNX policies and drive joint targets (`q`, `dq`, `tau`,
  `Kp`, `Kd`) at real-time control rates.
- Four checked-in ONNX weight files (see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §2 and
  [`MODEL_CARD.md`](MODEL_CARD.md)) that determine the robot's motion.

The security concerns for this repository are therefore both classical
software issues **and** robot-safety issues:

- Inadvertent disclosure of internal infrastructure (private IPs,
  service ports, credentials) baked into code, docs, configs, or
  captured media.
- Malicious or malformed ONNX / YAML files that could hijack inference
  or cause unbounded actuator commands.
- Supply-chain tampering of the ONNX weights, the LimX SDK submodule,
  or vendor wheels between publication and consumption.
- Unsafe motion commands reaching a physical robot when the operator
  did not expect real-hardware control (e.g. an example IP silently
  connecting to a lab robot).

Control-path *behavioral* safety (limits, watchdogs, e-stop, mounting
requirements) is described below under
[Real-hardware safety notice](#real-hardware-safety-notice); it is
part of the security envelope of this repository even though it is not
a classical CVE.

## Private-IP handling

`<robot-ip>` in this repository's Markdown / YAML command examples is
a **placeholder token**, not a real address. Substitute your own
robot or simulator IP before running. Nothing in this repository's
Python source, configuration, or CI hard-codes a private IP.

The internal-only sibling repository `tron2-rl-deploy-ros` retains a
documentation-example literal `10.192.1.2` in its source / launch
files (e.g. `Tron2HW.cpp`, `tron2_hw_node.cpp`, `tron2_hw.launch`),
kept per owner decision and declared in that repository's
`SECURITY.md`. That literal is documented there and is not mirrored
into this repository.

## Real-hardware safety notice

**This repository is not a simulation-only demo.** `main.py:20-27`
calls `robot.init(robot_ip)` on the LimX SDK, and
`controllers/SolefootController.py:34-50` (and the wheel-foot
equivalent) loads ONNX and writes joint targets that a real robot will
execute. Before running against real hardware:

1. **Suspend / hang the robot** on its mount (as noted in the README's
   real-world deployment section). Do not run a freshly-cloned
   controller with the robot on the ground.
2. **Confirm the target IP.** `main.py` requires the robot / SDK
   target IP as an explicit command-line argument
   (`python3 main.py <robot-ip>`). The `<robot-ip>` token that
   appears in this repository's docs is a **placeholder** — always
   pass your own robot's IP explicitly rather than relying on any
   documented example or default being appropriate for your
   environment. Nothing in this repository's source hard-codes a
   private IP; the internal-only sibling `tron2-rl-deploy-ros`
   retains a documentation-example literal `10.192.1.2` in its
   source / launch files (declared in that repo's `SECURITY.md`)
   and this repository intentionally does not.
3. **Verify the model files** match your robot variant. Loading an
   `SF_TRON2A` policy against a `WF_TRON2A` chassis (or vice-versa)
   will produce invalid joint commands.
4. **Have a physical e-stop / power cut-off** within reach for the
   entire session.
5. Prefer simulation first — bring the controller up against the
   sibling MuJoCo simulator before pointing it at a physical robot.

## Supported versions

Only the tip of the `main` branch and the most recent tagged release
receive security fixes. Older tags are provided as-is.

| Version    | Supported |
|------------|-----------|
| `main`     | ✅        |
| Latest tag | ✅        |
| Older tags | ❌        |

## Reporting a vulnerability

**Do not** open a public issue for security reports.

Email: **contact@limxdynamics.com**
Subject prefix: `[tron2-rl-deploy-python]`

Please include:

- Affected file(s) and commit / tag.
- A minimal reproducer or proof of concept.
- Impact assessment (e.g., "malformed `params.yaml` bypasses joint
  clamp", "controller connects to arbitrary IP without confirmation",
  "ONNX file hash mismatch not detected").
- Whether the report involves **real-hardware behavior** — if so, we
  will treat it as a robot-safety report and involve the robotics /
  safety owners in triage.
- Your preferred disclosure timeline and contact.

We aim to acknowledge reports within **3 business days** and provide a
remediation plan or an initial mitigation within **14 calendar days**.
We support coordinated disclosure; please do not publish details until
a fix or advisory is available.

## Out of scope

- Bugs in third-party components (`onnxruntime`, `numpy`, `scipy`,
  `pyyaml`, `pygame`, the LimX SDK wheel) — report those upstream.
- Requests to publish training data, training checkpoints, factory
  calibration, or firmware — this repository intentionally excludes
  those.
- Requests to change the SDK submodule pin without vendor sign-off —
  this is a governance question, not a security report.

## Safe harbor

Good-faith security research that follows this policy will not be
pursued legally by LimX Dynamics. Please respect user privacy, avoid
service disruption, and **do not test attacks against a physical robot
you do not own or are not explicitly authorized to operate**.
