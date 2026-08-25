# Prefill Economics Simulator

`asic-sim-prefill` is a deliberately simple analytical model for the question that matters before RTL: **what prefill throughput must a specialized accelerator deliver, and how cheap must it be, to beat a current GPU node on input-token economics?**

It is not a kernel benchmark and it is not a silicon-performance claim.

## Default experiment

```bash
asic-sim-prefill --model kimi-k3 --tokens 32768
```

The default hypothetical candidate is:

- 4 POPS low-precision matrix compute;
- 2 TB/s external-memory bandwidth;
- 2,048 GB capacity;
- 45% effective linear-GEMM utilization;
- 30% effective attention utilization;
- 75% effective memory-bandwidth utilization;
- 800 W assumed power.

The default reference is an 8x NVIDIA HGX B300 node.

## Compute sweep

```bash
asic-sim-prefill \
  --model kimi-k3 \
  --tokens 32768 \
  --candidate-bandwidth-tb-s 2 \
  --sweep-compute 1 2 4 8 16 32 64
```

This prints candidate input-token throughput, the current bottleneck, and three CAPEX ceilings:

- `1x`: maximum candidate purchase price for parity in input tok/s per euro;
- `3x`: maximum purchase price for a 3x advantage;
- `5x`: maximum purchase price for a 5x advantage.

That makes the rejection criterion explicit. If the physical BOM cannot fit below the 3x/5x ceiling, the architecture is not interesting enough to compensate for custom-silicon and software risk.

## Candidate price / TCO

```bash
asic-sim-prefill \
  --model kimi-k3 \
  --tokens 65536 \
  --candidate-compute-pops 16 \
  --candidate-bandwidth-tb-s 4 \
  --candidate-price-eur 12000 \
  --candidate-power-w 1200 \
  --utilization 0.60 \
  --lifetime-years 3 \
  --electricity-eur-kwh 0.15
```

When a candidate price is supplied, the simulator reports relative input tok/s/€ and amortized-plus-energy cost per million input tokens.

## Reference systems

### `b300-eu`

- Price: **€495,859.52** starting price.
- Date captured: **2026-08-25**.
- Price source: Supermicro Europe, preconfigured 8U HGX B300 8-GPU system.
- Compute model: 108 PFLOPS dense FP4 platform peak from NVIDIA HGX B300 specifications.
- Memory model: 64 TB/s aggregate HBM bandwidth research point (8 x ~8 TB/s).
- Power model: 14.5 kW published DGX B300 system consumption.

Price source:
https://store.supermicro.com/nl_es/servers/gpu.html?system_gpu_family=1556&system_gpu_model=1735

NVIDIA platform specification:
https://www.nvidia.com/es-es/data-center/dgx-b300/

### `mi355x-eu`

- Price: **€648,587 ex VAT** asking price.
- Date captured: **2026-08-25**.
- Price source: ServerMall EU, Gigabyte G893-ZX1-AAX4 with 8x MI355X.
- Compute model: 80.5 PFLOPS MXFP4 aggregate theoretical peak.
- Memory model: 64 TB/s aggregate HBM3E bandwidth.
- Power model: 11.2 kW accelerator TBP only (8 x 1.4 kW); host/cooling overhead is not included.

Price source:
https://servermall.com/sets/file-servers/?PAGEN_3=6&PAGEN_7=2

AMD specification:
https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html

The MI355X price is an observed reseller asking price, not an AMD MSRP. It should be treated as a market datapoint rather than a canonical price.

## Work model

Linear/MLP/projection work is approximated as:

```text
2 * active_parameters * input_tokens
```

For MoE models this deliberately uses active parameters per token rather than total resident parameters.

Explicit quadratic attention work is added as:

```text
4 * full_attention_layers * N^2 * hidden_size
```

Current v0 attention-layer mappings are intentionally narrow:

- Kimi K3: 24 full-attention/MLA layers;
- Qwen3.8-27B: 16 full-attention layers;
- dense models: all layers;
- GLM MoE presets: quadratic attention contribution omitted rather than inventing a DSA kernel model.

## Memory model

For long prefill, weights are assumed to benefit from reuse across the prompt. External weight traffic therefore starts from one resident-model pass per prompt, multiplied by `--weight-reload-factor` (default 1.15), instead of charging active weight bytes for every input token.

Activation traffic is represented by a configurable multiplier over hidden-state bytes across layers. KV output traffic is optional through `--kv-bytes-token`; it defaults to zero because the repository does not yet contain a sufficiently trustworthy generic KV-layout model across MLA, GQA, KDA and DSA architectures.

This is an intentionally optimistic roofline. A later simulator should replace these factors with operation-level tiled traces and measured kernels.

## Primary interpretation

The most useful number is not the candidate's theoretical POPS. It is the **CAPEX ceiling**.

If a 4-POPS candidate delivers only ~5% of the reference input-token throughput, its parity CAPEX ceiling is also only ~5% of the reference purchase price. A custom card must land materially below that ceiling before NRE, compiler, manufacturing and deployment risk make sense.

## Research rule

Do not tune assumptions to rescue an architecture. Sweep them. If reasonable pessimistic assumptions erase the economic advantage, reject the architecture early.
