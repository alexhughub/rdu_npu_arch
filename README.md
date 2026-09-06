# GPU-centric SoC UVM Testbench Reference

Default mode is CPU bypass: `+CPU_MODE=BYPASS`. The host AXI agent is active and UVM emulates host runtime/driver submission. CPU-inclusive E2E uses `+CPU_MODE=BOOT +BOOT_IMAGE=<image>`; the host AXI agent becomes passive and observes the real CPU-generated control traffic. Use a build-time define only when the project elaborates a physically different CPU-less/full-SoC netlist.

## Coding style
- Class/module/interface member/global state uses `m_`.
- Function/task/method local variables do not use `m_`.
- `parameter` and `localparam` do not use `m_`.
- External DUT protocol port names remain spec-facing names; DV interface storage signals use `m_`.

## Main verification flow
Boot/reset -> CSR/MMIO -> SQ/CQ/NOP -> HBM/memcpy -> tiny kernel -> synthetic command -> vendor replay -> LLM Prefill -> Decode -> dynamic batching -> performance/stress -> limited CPU-boot E2E.

The included protocol drivers are reference placeholders. Production projects should plug in commercial/internal AXI/APB/HBM/CHI/SMMU VIP while preserving the env/sequence/checker architecture.
