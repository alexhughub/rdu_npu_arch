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

   1. rdu_vs_npu_co_design_sweeper/co_design_sweeper.py ? High-level Python extreme-regimes
      analytical simulator.
   2. rdu_vs_npu_co_design_sweeper/co_design_python_sweep_results.csv ? Output dataset of the
      Python simulation runs.
   3. rdu_vs_npu_co_design_sweeper/RDU_vs_NPU_CoDesignSimulator.hpp ? Unified header defining
      both NPU and RDU.
   4. rdu_vs_npu_co_design_sweeper/RDU_vs_NPU_CoDesignSimulator.cpp ? C++ implementation of
      cycle-approximate pipelines, bank conflicts, global bus port contentions, and NoC credit
      queue backpressures.
   5. rdu_vs_npu_co_design_sweeper/run_co_design_sweeps.cpp ? Main C++ sweeps driver for the
      training vs serving operating regimes.
   6. rdu_vs_npu_co_design_sweeper/co_design_regime_sweep_results.csv ? C++ sweep dataset of
      the simulated extreme corners.
   7. rdu_vs_npu_co_design_sweeper/RDU_VS_NPU_CO_DESIGN_COMPARISON_REPORT.md ? Head-to-head
      operating regimes report.
   8. rdu_vs_npu_co_design_sweeper/RDU_VS_NPU_PYTHON_VS_CPP_REPORT.md ? Detailed report
      analyzing C++ vs. Python differences and their direct impact on Verilog/SystemVerilog RTL
      design.
   9. rdu_vs_npu_co_design_sweeper/Makefile ? High-optimization C++ compiler script.

