# Assumptions and provenance

This file keeps research assumptions separate from measured facts.

## Kimi K3

Moonshot AI publishes 2.8T total / 104B active parameters, 93 layers with one dense layer, hidden size 7168, 896 experts and top-16 routing, with a 1,048,576-token context. The released checkpoint reports MXFP4 weights / MXFP8 activations, while its config excludes several module classes from the MXFP4 target set.

Sources:

- https://huggingface.co/moonshotai/Kimi-K3
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json

`--bits N` therefore means an average architectural weight bit-width; it is not a claim about exact checkpoint bytes.

## GLM-5.2

The Z.ai/THUDM configuration is 744B total / ~40B active, 78 layers (3 dense + 75 MoE), hidden size 6144, 256 routed experts, top-8, one shared expert, and MoE intermediate size 2048.

Sources:

- https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- https://github.com/THUDM/slime/blob/main/scripts/models/glm5.2-744B-A40B.sh

## GLM-5.3

Z.ai states GLM-5.3 uses the same base model as GLM-5.2 and changes post-training only. Until the GLM-5.3 weights/config are published, the preset intentionally aliases GLM-5.2's physical architecture.

Source: https://z.ai/blog/glm-5.3

## Hardware presets

`hbm4-illustrative` (192 GB / 18 TB/s) is a generic comparison point, not a specific shipping GPU.

`raptor-like` (32 GB / 105 TB/s) approximates the public 3D-DRAM research point discussed around d-Matrix Raptor. It is not a microarchitectural reproduction.

The distributed fabrics are hypotheses: 32 GB/tile, 105 TB/s local bandwidth/tile, assumed 2 TB/s NoC links and 5 ns routers, arranged as 4x8 or 8x8 meshes.

M0 uses aggregate bandwidth only for ideal steady-state throughput. Single-stream latency uses local tile bandwidth.

## M1 decomposition

For equal-sized routed experts:

```text
total  = always_on + routed_pool
active = always_on + routed_pool * top_k / num_experts
```

Solving these equations reproduces the published total/active counts. For GLM-5.2, the inferred expert shard is ~37.55M parameters; the independent gated-FFN shape `3 * 6144 * 2048` is ~37.75M, providing a useful sanity check.

For Kimi K3 this count-derived method avoids assuming its Latent MoE is a conventional FFN.
