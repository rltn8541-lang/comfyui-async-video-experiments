"""Experimental orchestration contract. Requires isolated GPU worker adapters.

Not connected to ComfyUI. No GPU operations occur in this module itself.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence


class Mode(Enum):
    INDEPENDENT = 'independent'
    TAIL_FIRST = 'tail_first'
    TAIL_FIRST_CORRECTED = 'tail_first_corrected'


@dataclass(frozen=True)
class Clip:
    index: int
    prompt: str
    seed: int


class WorkerAdapter(Protocol):
    """Methods must execute GPU work in separate processes, not global device threads.

    Sample must return only after the sample is complete. Intermediate transfers
    and resource ownership are the adapter's responsibility. Tail decode must
    produce a raw frame before full decode starts and use a model-validated
    temporal receptive field. The adapter must surface cancellation and failures.
    """
    def begin(self, retain_model: bool) -> None: ...
    def sample(self, clip: Clip, reference: Any) -> Any: ...
    def decode_tail(self, latent: Any, clip: Clip) -> Any: ...
    def transfer_raw_reference(self, frame: Any, clip: Clip) -> Any: ...
    def decode_full(self, latent: Any, clip: Clip) -> Any: ...
    def correct_boundary(self, frames: Any, previous_tail: Any, clip: Clip) -> Any: ...
    def postprocess(self, frames: Any, clip: Clip) -> Any: ...
    def end(self) -> None: ...


def run_experiment(adapter: WorkerAdapter, clips: Sequence[Clip], *, mode: Mode,
                   initial_reference: Any = None, retain_model: bool = False,
                   cancelled=lambda: False) -> list[Any]:
    """Orchestrate work; adapters must implement GPU isolation and ownership.

    At most one full decode is in progress. Cancellation is cooperative and
    adapters must interrupt their own in-flight GPU commands. Exceptions are
    propagated and `end` runs after the decode thread has stopped.
    """
    if not isinstance(mode, Mode):
        raise ValueError('Invalid experiment mode')
    if not clips:
        raise ValueError('At least one clip is required')
    if [c.index for c in clips] != list(range(len(clips))):
        raise ValueError('Clip indices must be consecutive and start at zero')
    results = []
    previous_raw = None
    reference = initial_reference
    pending = None
    adapter.begin(retain_model)
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-decode') as pool:
            for clip in clips:
                if cancelled():
                    raise RuntimeError('Experiment cancelled')
                latent = adapter.sample(clip, reference)
                # Do not start a new GPU-2 operation while an earlier one runs.
                # Sampling above overlaps the previous full decode and postprocess.
                if pending is not None:
                    results.append(pending.result())
                if cancelled():
                    raise RuntimeError('Experiment cancelled')
                boundary_reference = previous_raw
                if mode is not Mode.INDEPENDENT:
                    raw = adapter.decode_tail(latent, clip)
                    reference = adapter.transfer_raw_reference(raw, clip)
                    previous_raw = raw

                def finish(latent=latent, clip=clip, boundary_reference=boundary_reference):
                    frames = adapter.decode_full(latent, clip)
                    if mode is Mode.TAIL_FIRST_CORRECTED and boundary_reference is not None:
                        frames = adapter.correct_boundary(frames, boundary_reference, clip)
                    return adapter.postprocess(frames, clip)

                pending = pool.submit(finish)
            if pending is not None:
                results.append(pending.result())
        return results
    finally:
        adapter.end()
