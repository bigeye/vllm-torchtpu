# SPDX-License-Identifier: Apache-2.0
"""Per-step scheduler trace (opt-in, host only).

Set ``VLLM_TPU_STEP_TRACE_DIR`` to a directory and the engine-core process
records what every ``Scheduler.schedule()`` call scheduled: per request, the
tokens scheduled this step, the tokens already computed before it (its KV
length), its prompt length, its prefix-cache hit and its spec-decode drafts,
plus the running and waiting queue sizes. Nothing touches the device.

Lines are buffered in memory and written as one JSONL part file every
``VLLM_TPU_STEP_TRACE_FLUSH_STEPS`` steps (default 5000), opened and closed
each time, so a server killed at the end of a benchmark loses at most the last
part. Read from ``os.environ`` rather than ``vllm_torchtpu.envs``: registered
env vars feed the compile-cache key, and tracing must not change it.

Line format: ``{"s": step, "t": unix_s, "run": n_running, "wait": n_waiting,
"r": [[scheduled, computed_before, prompt_len, cached_prefix, drafts], ...]}``.
"""

import atexit
import functools
import json
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


class _StepTraceWriter:

    def __init__(self, directory: str, flush_steps: int):
        self.directory = directory
        self.flush_steps = flush_steps
        self.prefix = f"steps-{os.getpid()}-{int(time.time())}"
        self.lines: list[str] = []
        self.part = 0
        self.step = 0
        self.failed = False
        os.makedirs(directory, exist_ok=True)
        atexit.register(self.flush)

    def record(self, scheduler, output) -> None:
        reqs = []
        drafts = getattr(output, "scheduled_spec_decode_tokens", None) or {}
        for req_id, n in output.num_scheduled_tokens.items():
            req = scheduler.requests.get(req_id)
            if req is None:
                continue
            # schedule() has already advanced num_computed_tokens by n.
            computed = req.num_computed_tokens - n
            reqs.append([n, computed, req.num_prompt_tokens,
                         max(getattr(req, "num_cached_tokens", 0), 0),
                         len(drafts.get(req_id, ()))])
        self.lines.append(json.dumps(
            {"s": self.step, "t": round(time.time(), 4),
             "run": len(scheduler.running), "wait": len(scheduler.waiting),
             "r": reqs},
            separators=(",", ":")))
        self.step += 1
        if len(self.lines) >= self.flush_steps:
            self.flush()

    def flush(self) -> None:
        if not self.lines:
            return
        path = os.path.join(self.directory, f"{self.prefix}-{self.part:05d}.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        self.lines = []
        self.part += 1


def install() -> None:
    directory = os.environ.get("VLLM_TPU_STEP_TRACE_DIR")
    if not directory:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.schedule, "_tpu_step_trace_patch", False):
        return
    writer = _StepTraceWriter(
        directory, int(os.environ.get("VLLM_TPU_STEP_TRACE_FLUSH_STEPS", "5000")))
    original_schedule = Scheduler.schedule

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        output = original_schedule(self, *args, **kwargs)
        if not writer.failed:
            try:
                writer.record(self, output)
            except Exception:  # never let tracing break serving
                writer.failed = True
                logger.exception("Step trace disabled after a recording error.")
        return output

    schedule._tpu_step_trace_patch = True
    Scheduler.schedule = schedule
    logger.info("Step trace on: writing scheduler steps to %s every %d steps.",
                directory, writer.flush_steps)
