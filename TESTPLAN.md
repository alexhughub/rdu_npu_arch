# GPU-centric SoC Test Plan

## Bring-up
1. Reset/boot/clock/ready and CSR reset values.
2. CSR/MMIO access, RW/RO/W1C, decode and side effects.
3. Queue configuration, NOP, SQ head/tail/doorbell, completion/IRQ.
4. HBM read/write/memcpy; burst, alignment, page/channel boundary.
5. SMMU/address translation positive and fault cases (project-specific agent hook).
6. Illegal opcode/descriptor/address and recovery.
7. Tiny vector/GEMM kernel functional smoke.

## Main GPU workloads
- Synthetic: DV-built descriptor/SQ plus vendor or DV test kernel.
- Replay: vendor command trace/descriptors/kernel binaries/memory images/golden output.
- LLM Prefill: model/input preload, KV generation, logits/output, Prefill perf.
- LLM Decode: KV read/append, token/logit checking, latency/bandwidth.
- Continuous batching: asynchronous request arrival, Prefill/Decode overlap, EOS removal, KV ownership.

## Checkers
Submission/head-tail/doorbell; descriptor; memory range/ownership; completion; KV structural/data; request-state scoreboard; numerical/token (project hook); Transformer Prefill roofline/performance.
