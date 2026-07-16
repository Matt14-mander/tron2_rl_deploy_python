<!--
Thanks for contributing to tron2-rl-deploy-python!
Please fill in the sections below. Delete any that are not applicable.
-->

## Summary

<!-- One paragraph: what and why. -->

## Type of change

- [ ] `fix`     — corrects a defect in the SF / WF controller or entry point
- [ ] `feat`    — new capability (dry-run mode, safety clamp, etc.)
- [ ] `model`   — adds / replaces an ONNX weight file **(requires provenance — see below)**
- [ ] `docs`    — README, THIRD_PARTY_NOTICES, MODEL_CARD, or CONTRIBUTING
- [ ] `ci`      — GitHub Actions or verification tooling
- [ ] `chore`   — repo maintenance (deps, formatting, cleanup)

## Affected variants

- [ ] `SF_TRON2A`
- [ ] `WF_TRON2A`
- [ ] Meta / repo-wide

## Verification

Paste the output (or a summary) of the local verification steps from
`CONTRIBUTING.md#verification-before-opening-a-pr`:

```text
python -m compileall: ...
ruff check: ...
import controllers: ...
yaml sanity: ...
onnx.checker: ...
license / TODO scan: ...
```

Additionally, if this PR affects control-loop behavior:

- [ ] Tested against the MuJoCo simulator (`127.0.0.1`) before any
      real-hardware run.
- [ ] Any real-hardware test was performed with the robot suspended /
      mounted (see `SECURITY.md`).

## Provenance & licensing

<!-- Required if the PR touches ONNX weights, params.yaml, or the SDK
     submodule. -->

- [ ] **No new model weights** are added by this PR, **OR** every new
      `*.onnx` has a matching row in `THIRD_PARTY_NOTICES.md` §2 and a
      completed section in `MODEL_CARD.md` (checkpoint id, training
      run, training data, evaluation, intended / out-of-scope use,
      redistribution status).
- [ ] `THIRD_PARTY_NOTICES.md` is up to date for any new / changed
      runtime dependency.
- [ ] The `limxsdk-lowlevel` submodule pin is **not** changed, **OR**
      the change has written SDK owner sign-off attached to the PR.
- [ ] `doc/*` media (if changed) have been EXIF-stripped and
      reviewed for people / office / non-public product visibility.

## Excluded artifacts

- [ ] This PR does **not** add: PyTorch checkpoints (`.pt`, `.pth`,
      `.ckpt`, `.safetensors`), SDK binaries (`.so`, `.dll`, `.dylib`,
      `.lib`, `.whl`), factory calibration values, firmware, rosbags /
      MCAP, or customer data.
- [ ] This PR does **not** hard-code credentials, API keys, tokens,
      or new private IPs / hostnames.

## Checklist

- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`.
- [ ] All commits are DCO-signed (`git commit -s`).
- [ ] CI is expected to pass.

## Related issues

<!-- Fixes #123 / Refs #456 -->
