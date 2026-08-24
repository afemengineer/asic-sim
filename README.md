# asic-sim

Fast, multi-fidelity architecture simulator for memory-stationary LLM inference accelerators.

The initial research target is a distributed compute + 3D-DRAM fabric: model weights remain local to memory-compute tiles while activations traverse a configurable on-chip network.

> Status: research prototype. Numbers are architectural estimates, not silicon claims.
