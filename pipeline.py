"""Experimental bounded two-stage pipeline; no ComfyUI or GPU integration yet."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque


def run_pipeline(items, sample, decode, *, cancelled=lambda: False, max_inflight=2):
    """Run serial sampling and serial decoding concurrently, returning ordered results.

    Callbacks own their GPU resources and must be safe in different threads. A
    bounded pending decodes limit intermediate memory; this is not a GPU scheduler.
    Cancellation prevents new work, but callbacks must implement their own interrupt.
    """
    if max_inflight < 1:
        raise ValueError('max_inflight must be at least one')
    pending = deque()
    results = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='experimental-decode') as executor:
        for item in items:
            if cancelled():
                raise RuntimeError('cancelled')
            while len(pending) >= max_inflight:
                results.append(pending.popleft().result())
            intermediate = sample(item)
            if cancelled():
                raise RuntimeError('cancelled')
            pending.append(executor.submit(decode, intermediate))
        while pending:
            results.append(pending.popleft().result())
    return results
