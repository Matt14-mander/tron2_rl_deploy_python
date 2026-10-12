"""Observation-only PAINT adapter; never supplies policy wrench or commands."""

import json
import copy
import queue
import threading
from pathlib import Path

import numpy as np


class IntentObserver:
    def __init__(self, model_dir, log_path):
        import torch
        self.torch = torch
        directory = Path(model_dir)
        self.meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "schema": "paint_pd_response_v1",
            "joint_names": [f"arm{i}_Joint" for i in range(1, 7)],
            "output_frame": "payload_local", "output_point": "payload_COM",
            "output_order": ["Fx_N", "Fy_N", "Mz_Nm"],
            "input_units": "rad, rad/s, rad",
            "output_semantics": "external_on_payload during preceding control interval",
        }
        for name, value in expected.items():
            if self.meta.get(name) != value:
                raise ValueError(f"Intent metadata mismatch: {name}")
        self.history = self.meta["history"]
        if (type(self.history) is not int or self.history < 1
                or self.meta.get("input_size") != self.history * 18
                or not np.isclose(self.meta["step_dt"], .02)):
            raise ValueError("Intent requires 50 Hz q/dq/executed-target history")
        self.net = torch.jit.load(str(directory / "intent_estimator.ts"), map_location="cpu").eval()
        with torch.no_grad():
            probe = self.net(torch.zeros(1, self.history * 18))
        if tuple(probe.shape) != (1, 3) or not torch.isfinite(probe).all():
            raise ValueError("Invalid intent output")
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = path.open("x", encoding="utf-8")
        self.rows = []
        self.last_time = None
        self.last_state_count = None
        self.log.write(json.dumps({"type": "contract", "metadata": self.meta,
            "mode": "observation_only", "timing": "host reception; SDK execution not acknowledged",
            "arm_pd_backend": "explicit_MIT", "policy_wrench_source": "OCS2_or_zero"}) + "\n")

    def update(self, now, state_count, state_stamp, q, dq, target, solution):
        reason = None
        if self.last_time is None:
            reason = "initial"
        elif not .018 <= now - self.last_time <= .022:
            reason = "sample_time_gap"
        if state_count == self.last_state_count or state_stamp is None or now - state_stamp > .02:
            reason = "stale_or_repeated_state"
        if target is None:
            reason = "no_previous_command"
        if reason:
            self.rows.clear()
        prediction = None
        if reason not in ("stale_or_repeated_state", "no_previous_command"):
            row = np.concatenate([q[10:16], dq[10:16], target]).astype(np.float32)
            if row.shape != (18,) or not np.isfinite(row).all():
                raise ValueError("Invalid intent telemetry")
            self.rows.append(row)
            self.rows = self.rows[-self.history:]
            if len(self.rows) == self.history:
                with self.torch.no_grad():
                    result = self.net(self.torch.from_numpy(np.concatenate(self.rows)[None])).numpy()[0]
                if not np.isfinite(result).all():
                    raise RuntimeError("Nonfinite intent estimate")
                prediction = result.tolist()
        self.last_time, self.last_state_count = now, state_count
        self.log.write(json.dumps({"type": "sample", "host_time": now,
            "state_age": None if state_stamp is None else now - state_stamp,
            "q": q[10:16].tolist(), "dq": dq[10:16].tolist(),
            "previous_sent_position": None if target is None else target.tolist(),
            "valid": prediction is not None, "history_reset_reason": reason,
            "interaction_wrench_payload_local": prediction,
            "preceding_future_wrench": solution.wrench_prediction.tolist(),
            "arm_velocity_target": solution.arm_velocity.tolist(),
            "arm_effort_target": solution.arm_effort.tolist(),
            "base_command": solution.base_command.tolist(),
            "pd_only_training_contract": bool(np.allclose(solution.arm_velocity, 0)
                                             and np.allclose(solution.arm_effort, 0)),
        }) + "\n")
        self.log.flush()

    def close(self):
        self.log.close()


class ObservationClock:
    """Keep sampling phase after a late tick; never fabricate catch-up frames."""
    def __init__(self, start, period=.02):
        self.next_time = start
        self.period = period

    def due(self, now):
        if now < self.next_time:
            return False
        missed = int((now - self.next_time) // self.period)
        self.next_time += (missed + 1) * self.period
        return True


class AsyncIntentObserver:
    """Bounded FIFO: copy telemetry in control thread, infer/write in worker."""
    def __init__(self, model_dir, log_path):
        self.observer = IntentObserver(model_dir, log_path)
        self.queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.error = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, name="intent-observer", daemon=True)
        self.thread.start()

    def update(self, now, state_count, state_stamp, q, dq, target, solution):
        if self.error is not None:
            raise RuntimeError("Intent observer worker failed") from self.error
        item = (now, state_count, state_stamp, q.copy(), dq.copy(),
                None if target is None else target.copy(), copy.deepcopy(solution))
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            # The next accepted sample retains its actual timestamp, so the
            # existing gap check breaks history across dropped samples.
            self.dropped += 1

    def _run(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                self.observer.update(*item)
            self.observer.log.write(json.dumps({"type": "observer_summary",
                "queue_dropped_samples": self.dropped}) + "\n")
        except Exception as error:
            self.error = error
            print(f"[ERROR] Intent worker: {error}", flush=True)
        finally:
            self.observer.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5.)
        print(f"[INFO] Intent worker: dropped={self.dropped}, "
              f"stopped={not self.thread.is_alive()}, error={self.error}", flush=True)
