---
name: Bug report
about: Controller / policy inference / SDK integration defect
title: "[bug] <short summary>"
labels: bug
assignees: ''
---

## Affected component(s)

<!-- Path within the repo, e.g. controllers/SolefootController.py -->

- File(s):
- Variant(s): `SF_TRON2A` / `WF_TRON2A` / `DASF_TRON2A` (pick)
- Commit / tag:

## Environment

- Python version:
- OS + arch (`uname -m` / distro):
- LimX SDK wheel version (from `pip show limxsdk`):
- `onnxruntime` version (CPU vs GPU):
- Target: MuJoCo simulator (`127.0.0.1`) / real robot / other:

## Expected behavior

<!-- What the controller should do (velocity tracking, safe stop, etc). -->

## Actual behavior

<!-- What you observed: log lines, joint diagrams, stack traces.
     If this involves real-hardware motion, describe the mount / suspension
     state and whether an e-stop was triggered. -->

## Minimal reproduction

```bash
# commands that reproduce it locally, ideally in simulation first
export ROBOT_TYPE=SF_TRON2A  # or WF_TRON2A / DASF_TRON2A
python3 main.py
```

## Additional context

<!-- Cross-links to related issues, sim-vs-real divergence, joystick input. -->

## Checklist

- [ ] I have searched existing issues.
- [ ] I have included the exact commit / tag.
- [ ] If this involves real-hardware behavior, the robot was suspended
      / mounted for the test.
- [ ] I am **not** reporting a security issue or robot-safety incident
      (those go to `contact@limxdynamics.com` per `SECURITY.md`).
