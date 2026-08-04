# 1000-TOPS NPU Low-Level C++ Cycle-Approximate Simulator

This directory contains our low-level, high-performance **C++ NPU Cycle-Approximate Simulator**. It models:
1. **Systolic Shifting Stages:** Data propagation and setup cycle-by-cycle bubbles across a 2D PE grid.
2. **Central SRAM Port Contention:** Models arbitration conflicts and pipeline stalls when loading inputs or writing back outputs through shared ports.
3. **MoE Expert Weight Thrashing:** Models cycle-approximate memory thrashing where dynamic experts must load weights from HBM, adding queuing delays.
4. **Global Bus Wire-Charging Capacitance:** Simulates energy based on monolithic bus wire lengths.

---

## Compilation & Execution

To compile the C++ simulator:
```bash
make
```

To run the sweeps:
```bash
./run_sweeps
```
This will:
1. Sweep **81 centralized systolic configurations**.
2. Output results to `npu_1000tops_cpp_sweep_results.csv`.
3. Generate a co-design report in `NPU_1000TOPS_CPP_CO_DESIGN_REPORT.md` analyzing the low-level bus contentions and systolic shifting registers.
