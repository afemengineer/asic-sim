# asic-sim

Fast, multi-fidelity architecture simulator for **memory-stationary LLM inference accelerators**.

The initial research target is a distributed compute + 3D-DRAM fabric: model weights remain resident beside compute tiles while token activations traverse an on-chip network. The project starts deliberately simple: reject bad architectural ideas in milliseconds before spending time on cycle-accurate simulation or RTL.

## Current scope: M0 + early M1

M0 is an analytical decode roofline. It models:

- model storage and quantization feasibility;
- active weight bytes per generated token;
- approximate MoE activation traffic between tiles;
- local-memory bandwidth limits;
- ideal NoC injection and router-latency floors;
- conservative single-stream decode latency;
- ideal steady-state pipelined throughput.

M1 also includes a metadata-only equal-expert decomposition derived from published total/active parameter counts and a balanced round-robin placement estimator. This is especially useful for Kimi K3 because it avoids pretending its Latent MoE is a conventional three-matrix FFN.

It **does not** yet model compute throughput, KV-cache traffic, bank conflicts, NoC contention/bisection bandwidth, thermals, packet overhead, exact expert placement, or real tensor-engine scheduling. Reported token rates are roofs, not product-performance claims.

## Built-in models

| Model | Total params | Active params | Layers | Experts | Top-k | Notes |
|---|---:|---:|---:|---:|---:|---|
| Kimi K3 | 2.8T | 104B | 93 | 896 | 16 | 1 dense + 92 MoE; 69 KDA + 24 gated MLA |
| GLM-5.2 | 744B | ~40B | 78 | 256 | 8 | 3 dense + 75 MoE; DSA/IndexShare |
| GLM-5.3 | 744B | ~40B | 78 | 256 | 8 | Same base model as GLM-5.2; post-training changes only |

Kimi K3 is intentionally included because it immediately stresses the capacity thesis. A naive 4-bit representation of 2.8T parameters is already 1.4 TB before quantization metadata or higher-precision exceptions, so the 32 x 32 GB / 1.024 TB fabric cannot hold it. The simulator calculates the exact average bit-width needed for any capacity target.

## Built-in hardware points

- `hbm4-illustrative`: 192 GB, 18 TB/s generic HBM comparison point.
- `raptor-like`: 32 GB, 105 TB/s single 3D-DRAM tile research reference.
- `fabric-32x32`: 32 x 32 GB = 1.024 TB, 105 TB/s local per tile, 2 TB/s mesh link.
- `fabric-64x32`: 64 x 32 GB = 2.048 TB for Kimi-K3-class capacity experiments.

The HBM and fabric values are explicit research assumptions, not claims about a shipping product. Change them aggressively.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest -q
```

## First commands

```bash
asic-sim list-models
asic-sim capacity --model kimi-k3 --capacity-gb 1024
asic-sim compare --model glm-5.2 --bits 4
asic-sim compare --model kimi-k3 --hardware fabric-64x32 --bits 4
asic-sim placement --model kimi-k3 --hardware fabric-64x32 --bits 4
asic-sim compare --model kimi-k3 --hardware fabric-32x32 --bits 2.75 --overhead 0.05
```

## Important interpretation

For a tiled fabric the simulator reports two different bandwidth roofs:

1. **Single-stream roof**: a conservative token follows sequential transformer dependencies and sees local tile bandwidth plus idealized NoC cost. It does not receive aggregate fabric bandwidth for free.
2. **Steady-state bandwidth roof**: assumes enough independent sequences to pipeline/load-balance work across all tiles. This may approach aggregate bandwidth, but is an upper bound until the event-driven simulator proves the mapping can sustain it.

That distinction is central to this project. Any architecture that only looks good by confusing aggregate bandwidth with single-token bandwidth should be rejected.

## Roadmap

### M1 — workload graph + placement

Already started: count-derived always-on/routed decomposition, balanced occupancy and expected remote-expert traffic. Next:

- materialize transformer layers as metadata-only operations;
- place individual expert/layer groups onto physical tiles;
- report exact per-tile memory occupancy and traffic;
- synthetic balanced / Zipf / hot-expert routing distributions;
- export traces for later NoC simulation.

### M2 — discrete-event NoC

- packet/event queue rather than per-cycle whole-model simulation;
- link contention, bisection limits, router queues and backpressure;
- replay real MoE routing traces;
- characterize ugly bursts instead of simulating trillions of clocks.

### M3 — external validation

- Timeloop/Accelergy for representative tile operations;
- BookSim for selected NoC traces;
- ASTRA-sim cross-checks where appropriate;
- calibrate assumptions against published silicon measurements.

### M4 — RTL / FPGA

Only architectures that survive M0-M3 deserve RTL time.

## Sources for built-in model metadata

- Kimi K3 model card/config: https://huggingface.co/moonshotai/Kimi-K3
- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2
- GLM-5.2 training configuration: https://github.com/THUDM/slime/blob/main/scripts/models/glm5.2-744B-A40B.sh
- GLM-5.3 announcement: https://z.ai/blog/glm-5.3

## Research rule

If a result looks spectacular, assume the simulator is wrong until an independent model reproduces it.
