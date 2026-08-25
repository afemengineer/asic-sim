# Prefill BOM Feasibility Gate

`asic-sim-prefill-bom` is the final rejection-stage check for the prefill hardware thesis. It takes the first design-space point that meets the default latency guardrail (64 POPS / 8 TB/s for Kimi K3 at 32K prompt length in the current analytical model) and asks whether a physical appliance can plausibly fit below the economic CAPEX ceilings.

This is deliberately a **cost floor**, not a product BOM quote.

## Run

```bash
asic-sim-prefill-bom --model kimi-k3 --tokens 32768
```

The default volume cases are:

- 100 appliances: current quoted GDDR7 price, no discount;
- 1,000 appliances: 20% assumed direct-volume discount;
- 10,000 appliances: 35% assumed direct-volume discount.

Those discounts are intentionally optimistic assumptions, not supplier quotes. Override them with repeated `--volume UNITS:MULTIPLIER` arguments.

## Why capacity becomes the physical problem

Kimi K3 at 4 bits plus 5% overhead requires:

```text
2.8T parameters * 4 / 8 * 1.05 = 1.47 TB
```

The concrete memory device in the default model is a 16 Gb (2 GB) x32 GDDR7 package at 28 Gb/s/pin. A current August 2026 market quote for Samsung K4VAF325ZC-SC28 is $46.50 per package.

That means:

```text
1.47 TB / 2 GB = 735 GDDR7 packages
```

The CLI groups 16 packages per module as an aggressive 512-bit-class GDDR interface:

```text
735 packages -> 46 modules
32 GB/module
1.792 TB/s raw bandwidth/module
~82.3 TB/s raw aggregate pin bandwidth
```

The target machine only needs 8 TB/s in the current prefill design point, so the GDDR array is massively over-provisioned for bandwidth **because capacity forces hundreds of packages to exist**.

This is the central physical result of the experiment.

## Default bottom-up assumptions

### Memory

- Device: Samsung K4VAF325ZC-SC28
- Density: 16 Gb / 2 GB
- Width: x32
- Data rate: 28 Gb/s/pin
- Raw bandwidth/device: 112 GB/s
- Current quoted price: $46.50/device
- Quote date captured: 2026-08-24

Quote:
https://microworkskorea.tistory.com/2659

Samsung specification:
https://semiconductor.samsung.com/us/dram/gddr/gddr7/k4vaf325zc-sc28/

Micron independently documents the same 16 Gb x32 / 28 Gb/s GDDR7 class and 112 GB/s per placement:
https://www.micron.com/products/memory/graphics-memory

### Logic silicon

Epoch AI estimates B200 logic fabrication at roughly $900 for two ~800 mm2 4NP compute chiplets. B200 provides about 10 PFLOPS dense FP4 per package.

The BOM gate grants the hypothetical inference-only ASIC a **2x compute-per-dollar specialization advantage** versus that baseline. This is intentionally favorable to the custom design.

Source:
https://epoch.ai/data-insights/b200-cost-breakdown

### Module auxiliary cost

Epoch AI estimates ~$480 per B200 module for VRMs, PCB and final module-level assembly/testing. The BOM gate reuses that as a per-module proxy, plus only $100 for a simple package and $100 for internal fabric per module.

This is not a quote for a 512-bit custom GDDR module. It is a floor assumption.

### NRE

Default NRE is $47M for a leading-edge 5 nm-class full-mask program. Public 2026 estimates vary substantially and leading-edge total NRE can exceed $100M.

Default reference:
https://siliconandsteel.co/tools/market-data/

Broader leading-edge range:
https://siliconanalysts.com/analysis/fabless-startup-tapeout-cost-guide

### External networking

The default appliance reserves only €1,990 for external high-speed networking, corresponding to a current ConnectX-8 800 Gb/s adapter price point. Internal module interconnect is separately represented by the deliberately small $100/module fabric allowance.

Reference:
https://www.snswitch.com/products/900-9x81e-00ex-st0

### FX

Default conversion is:

```text
1 USD = 0.85673 EUR
```

Captured 2026-08-25.

## What is still excluded

Even the `required sell price` output does not include several real commercial costs:

- software/compiler development beyond NRE;
- technical support;
- warranty and spares reserve;
- inventory financing;
- sales/channel costs;
- regulatory qualification;
- respin contingency;
- custom cooling engineering;
- datacenter integration labor.

The CLI does include a 10% variable-cost contingency and a 30% gross-margin requirement by default.

## Interpretation

The purpose is not to prove a product is buildable. It is to reject it when the *optimistic floor* already fails.

The useful questions are:

1. Does the required sell price fit below the 3x CAPEX ceiling?
2. Does it fit below the 5x ceiling?
3. At what production volume does NRE stop dominating?
4. Does the physical memory topology become absurd before economics even matter?

A design that only works at 10,000+ appliance volume is not equivalent to a startup-friendly hardware opportunity. At that volume the program implies millions of GDDR packages, a very large semiconductor commitment and hundreds of millions of euros of product revenue.

## Research rule

If the architecture needs lower component prices, higher utilization, better specialization density, lower NRE, lower margin and higher production volume **simultaneously** to survive, treat that as a failed hypothesis rather than a tuning opportunity.
