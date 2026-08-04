# Multi-Dimensional Hardware Co-Design Sweeper: RDU vs. NPU

This directory contains a low-level **C++ Structural Simulator** designed to contrast and evaluate **SambaNova Spatial RDU** vs. **TPU-style Centralized Systolic NPU** across two extreme, opposing operating regimes:

1. **Regime A: Large-Batch Training & Dense Serving** ($B=128$, $S=512$).
2. **Regime B: Real-Time Serving & Extreme Context** ($B=1$, $S=32,768$).

---

## Compilation & Execution

To compile:
```bash
make
```

To run the extreme regime co-design simulation sweeps:
```bash
./run_sweeps
```

This will:
1. Run structural co-design loops for LLaMA-3-70B and DeepSeek-V3 MoE under both operating corners.
2. Export the simulated database to `co_design_regime_sweep_results.csv`.
3. Compile the comprehensive co-design comparison report in `RDU_VS_NPU_CO_DESIGN_COMPARISON_REPORT.md`.
