# 1000-TOPS TPU-style Centralized Systolic NPU High-Level Simulator

This directory contains our high-level **TPU-style Centralized Systolic NPU Simulator**. It accepts configurable hardware properties and executes multi-dimensional parameter sweeps running **LLaMA-3-70B** and **DeepSeek-V3 MoE** layers.

---

## 1. Key NPU Hardware Variables Modeling
* **Central SRAM Capacity:** A single monolithic block (no local tile SRAM). No AGUs or on-chip compression pipelines.
* **Global Bus Interface Bandwidth:** Speed of the wide global routing bus connecting the PE array rows to the central SRAM.
* **PE Array Dimensions:** Sized up to $712 \times 712$ for 1000-TOPS peak performance.
* **MoE Expert Weight Thrashing:** Models weight thrashing due to sequential token execution (fetching routing experts dynamically over the HBM).

---

## 2. Execution

To run the high-level sweeps, execute:
```bash
python3 run_workload_sweeps.py
```
This will:
1. Sweep **81 unique hardware corners**.
2. Save the database to `npu_1000tops_sweep_results.csv`.
3. Generate a co-design analysis report in `NPU_1000TOPS_CO_DESIGN_REPORT.md` analyzing the Central SRAM activation spill wall and MoE weight thrashing.
