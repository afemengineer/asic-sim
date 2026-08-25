# Prefill Design-Space Sweep

`asic-sim-prefill-space` turns the single-point prefill roofline into a rejection-stage 2-D design-space study.

The command sweeps candidate low-precision compute against external-memory bandwidth and reports, for every grid point, the **maximum all-in purchase price of the complete candidate appliance** that still preserves a requested input-token-throughput-per-euro advantage over the selected GPU-node reference.

It is not a BOM estimate. It is an economic ceiling.

## Default Kimi K3 experiment

```bash
asic-sim-prefill-space --model kimi-k3 --tokens 32768
```

Defaults:

- compute grid: 1, 2, 4, 8, 16, 32, 64 POPS;
- bandwidth grid: 0.5, 1, 2, 4, 8, 16 TB/s;
- 2,048 GB candidate capacity;
- 3x required input tok/s/€ advantage;
- candidate linear-GEMM efficiency: 45%;
- candidate attention efficiency: 30%;
- candidate memory efficiency: 75%;
- 800 W candidate power assumption;
- B300 EU purchase reference;
- a latency guardrail that flags candidate single-prompt prefill slower than 2x the reference.

A `!` suffix on a CAPEX-ceiling cell means the economic ceiling exists mathematically, but the design point fails the latency guardrail.

## Example: stricter 5x target

```bash
asic-sim-prefill-space \
  --model kimi-k3 \
  --tokens 65536 \
  --target-advantage 5 \
  --compute-pops 2 4 8 16 32 64 \
  --bandwidth-tb-s 1 2 4 8 16
```

## Disable the latency guardrail

```bash
asic-sim-prefill-space --max-slowdown 0
```

This is useful for pure throughput/$ exploration, but it should not be used as the primary criterion for interactive agentic workloads.

## Knee reports

The command also prints two 95%-capture knee tables:

- **Memory knee**: at each compute point, the minimum bandwidth that reaches 95% of the best throughput available in the supplied bandwidth grid.
- **Compute knee**: at each bandwidth point, the minimum compute that reaches 95% of the best throughput available in the supplied compute grid.

These identify where buying more MACs or more bandwidth has stopped producing meaningful prefill throughput.

Change the capture threshold with:

```bash
asic-sim-prefill-space --knee-capture 0.90
```

## Interpretation

The intended rejection rule is simple:

1. Pick a required advantage, normally at least 3x because a custom accelerator carries NRE, software, manufacturing and integration risk.
2. Find grid points that meet the latency guardrail.
3. Read the maximum all-in CAPEX envelope for those points.
4. Compare that envelope with a physically plausible memory + silicon + board + networking + cooling BOM.
5. If the BOM cannot fit comfortably below the envelope, reject the architecture rather than tuning simulator assumptions to save it.

The most dangerous result is a tiny feasible island that requires simultaneously optimistic compute efficiency, memory efficiency, quantization, power and pricing. A broad viable region is much more credible than one heroic point.
