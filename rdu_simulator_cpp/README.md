# 1000-TOPS RDU Low-Level C++ Cycle-Approximate Simulator

This directory contains a low-level, high-performance **C++ RDU Cycle-Approximate Simulator**. It models:
1. **PCU Pipeline Stages:** (Fetch, Decode, Register Read, Tensor-MAC Execute, Writeback) and logs pipeline stall events and hardware clock bubbles.
2. **PMU Dual-Port 8T SRAM Bank Conflicts:** Models cycle-accurate structural hazards where PCU reads conflict with NoC prefetch writes, adding single-cycle delays.
3. **NoC Router Input Buffers and Flow Credits:** Models credit-based backpressure during dynamic Mixture-of-Experts (MoE) token dispatches on the 2D routing mesh.
4. **HBM Port Arbitration Queues:** Models DRAM queuing delays and memory controller scheduling.

---

## Compilation & Execution

To compile the C++ simulation suite, execute:
```bash
make
```

To run the multi-dimensional structural sweeps:
```bash
./run_sweeps
```

This will:
1. Run co-design sweeps over **81 physical configurations** (3 Grids x 3 SRAM sizes x 3 Compressions x 3 HBM speeds).
2. Save the complete simulated database to `rdu_1000tops_cpp_sweep_results.csv`.
3. Compile a high-fidelity co-design report in `RDU_1000TOPS_CPP_CO_DESIGN_REPORT.md` analyzing low-level clock stalls and routing backpressures.
