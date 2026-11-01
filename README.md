# ComfyUI asynchronous video experiments

Experimental nodes and an orchestration sketch for overlapping video sampling with decoding and post-processing on separate workers. This repository is a research starting point, not a ready-to-run multi-GPU application.

## Pipeline used in a production experiment

```text
CPU:   prepare text + reference for clip N+1  ──────────┐
GPU 1: sample clip N+1                       ──────────┤
GPU 2: decode + AI upscale clip N             ─────────┘
                                    → encode and join clips
```

Each worker runs in a separate process. Sampled latents are checkpointed before handoff, and finished clips can be reused after interruption. The original production driver is specific to a private deployment and is **not included**; the diagram describes its design, not functionality supplied by this repository. The included `experiment_runner.py` and `pipeline.py` define orchestration contracts with no ComfyUI worker adapter, device allocator, or GPU integration. They cannot independently generate video.

## Experimental ComfyUI nodes

Copy `comfy_nodes` to a dedicated directory inside `ComfyUI/custom_nodes/` and restart ComfyUI. Dependencies include PyTorch, ComfyUI, and safetensors. `ExperimentalSamplerProfile` additionally requires the separate `StudioSamplerCustomAdvanced` node, which is not included here. Other nodes are for last-frame partial VAE decoding, comparison, boundary blending, and MiniMax-style AV latent or conditioning handoff. Nodes are model- and ComfyUI-version-dependent; compatibility is not guaranteed. Review their inputs before use.

Partial temporal VAE decode can differ from full decode. Compare the result with `ExperimentalTailCompare` before relying on it for continuity. Pixel blending is a simple transition, not a motion-aware seam repair. Never use untrusted checkpoint files; conditioning may contain sensitive prompt-derived information.

## What was actually checked

On two Intel Arc A770 GPUs, isolated short MiniMax video experiments showed overlapping *worker intervals* for sampling and decoding, and separate independent clip workers. A small last-frame experiment produced a mean absolute difference of 0.013326 versus full decoding. These observations do not establish simultaneous GPU kernel execution or general performance. Tiny tests cannot establish reliability for all resolutions, models, failure modes, or devices. A longer private batch used the CPU/GPU pipeline above; interruptions required manual recovery. Automatic GPU allocation and seamless model retention remain unverified.

The included orchestration unit tests use mock workers; they do not test ComfyUI or GPUs. Run from this directory with `python -m unittest discover -s tests`. GPU/ComfyUI compatibility and end-to-end use must be verified in the target environment.

No models, game assets, prompts, generated videos, personal paths, or credentials are included. Model and third-party dependency licenses apply separately.
