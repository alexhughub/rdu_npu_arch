# 1000-TOPS SambaNova-style RDU Microarchitectural Simulator

This directory contains a highly configurable, microarchitectural and analytical simulator modeled specifically for a **1000-TOPS Reconfigurable Dataflow Unit (RDU)**. It allows hardware designers to explore the multi-dimensional co-design space by running massive next-generation Large Language Models (LLMs) like **LLaMA-3-70B** and **DeepSeek-V3/V4 (Mixture of Experts - MoE)**.

---

## 1. High-Level vs. Low-Level Simulator Modeling (Python vs. C++)

When designing microarchitectural simulators in industrial semiconductor groups (such as Intel, NVIDIA, or Google TPU teams), engineers divide simulators into two distinct tiers:

### Tier 1: High-Level Analytical & Architectural Simulators (Python)
* **What it is:** A Python-based simulator designed around first-principles analytical models, graph-level mappings, and coarse-grained memory-access trackers.
* **Why Python is Better:**
  * **Rapid Prototyping:** Exploring coarse trade-offs (e.g. "What happens if we double the HBM3 bandwidth from 2.4 to 4.8 TB/s?", "Does increasing SRAM from 128KB to 256KB prevent spilling?") can be implemented and run in seconds.
  * **Rich Scientific Ecosystem:** Leveraging libraries like `NumPy` and `Pandas` allows for fast matrix modeling, parameter sweeps, and instant data analysis.
  * **Rapidly Changing Workloads:** Adapting to new model architectures (like DeepSeek MoE or MLA) requires writing high-level topological rules rather than rebuilding complex structural objects.
* **Limitations:** Python cannot model cycle-by-cycle hardware states efficiently due to execution overhead. It cannot accurately capture sub-nanosecond wire delays, gate-level clock-edge bank conflicts, or queue-buffer credit-based flow control.

### Tier 2: Low-Level Cycle-Accurate & Cycle-Approximate Simulators (C++)
* **What it is:** A compiled C++ simulator modeled around cycle-by-cycle clock-edge transitions, pipeline stages, routing switch states, and hardware queues.
* **Why C++ is Better:**
  * **Performance & Scale:** Simulating billions of clock cycles on a 1000-tile chip requires massive execution speeds and highly optimized memory layouts. C++ is up to 100x faster than Python for structural, cycle-by-cycle loops.
  * **Structural Modeling:** C++ class structures can directly map to actual RTL blocks (e.g., modeling a 5-stage PCU execution pipeline, a credit-based 2D NoC routing switch, or an SRAM bank arbiter).
  * **Co-simulation:** Easy integration with hardware description language (HDL) environments (via SystemC or Verilator) for physical verification.
* **Limitations:** C++ is slow to write, has long compilation turnarounds, and rewriting compiler/workload mapping algorithms takes significantly more engineering time.

### Our Approach for This Simulator:
We build a **High-Level Architectural and Pipeline-Approximate RDU Simulator in Python**. This allows us to rapidly simulate massive workloads (like 70-Billion parameter models) and perform high-dimensional sweeps of HBM size, SRAM capacity, compression modes, and grid layouts, generating instant, publication-quality co-design insights.

---

## 2. RDU Simulator Architecture & Modeling Principles

The simulator represents a 2D mesh grid of independent **reconfigurable tiles**, consisting of:
1. **PCU (Pattern Compute Unit):** Houses SIMD vector lanes, instruction decoders, and a tensor-MAC engine.
2. **PMU (Pattern Memory Unit):** Distributed, dual-ported 8T SRAM bank with local Address Generation Units (AGUs) and stream compressors.
3. **NoC (Network-on-Chip) Router:** A 5-port 2D mesh router (North, South, East, West, Local) connecting tiles.

```
       +-----------------------------------+
       |            RDU TILE               |
       |  +---------+   NoC   +---------+  |
 <========> Router  |<=======>|  Router <========>
       |  +----+----+  Switch +----+----+  |
       |       |                   |       |
       |  +----+----+         +----+----+  |
       |  |   PCU   |         |   PMU   |  |
       |  | (ALUs)  |         | (SRAM)  |  |
       |  +---------+         +---------+  |
       +-----------------------------------+
```

### Key Parameters Configured in `config.ini`:
* **Grid Dimensions:** Size of the 2D mesh (e.g., $32 \times 32$ for 1024 tiles).
* **PMU SRAM Size:** Physical bytes of SRAM integrated into each tile (e.g., 128 KB or 256 KB).
* **Compression Support:** Toggle hardware low-overhead compression (FP8, INT4, or None).
* **HBM Bandwidth & Capacity:** The external high-bandwidth memory interface properties.
* **NoC Routing Link Bandwidth:** The maximum transfer rate between adjacent switches (GB/s).

---

## 3. Advanced Workload Modeling

The simulator supports running two major, state-of-the-art workloads:

### A. LLaMA-3-70B Layer
* Represents a massive, dense datacenter Transformer block.
* Activations are dense and sequential.
* Memory access is governed by loading heavy model weights and streaming activation tiles.

### B. DeepSeek-V3 MoE Layer (Mixture-of-Experts)
* Models a complex Mixture-of-Experts block.
* Contains a set of **shared experts** (always mapped on-chip) and **routed experts**.
* For each token, only a subset of routed experts (e.g., 6 out of 256) are activated.
* **Spatial Router Mapping:** We model mapping specific experts to specific regions of the 2D RDU mesh. When a token is processed, its activation vector is **dynamically routed across NoC links** to the active PCU tiles executing those expert weights.
* Models **NoC Routing Congestion**: If too many tokens route to the same expert tiles, the simulator models the queue stalls and link-congestion delays, predicting real-world MoE performance!

---

## 4. Execution

To execute the simulator and run architectural Sweeps:
```bash
python3 run_workload_sweeps.py
```
This script will:
1. Load configuration templates.
2. Simulate LLaMA-3-70B and DeepSeek-V3.
3. Sweep grid sizes, SRAM budgets, and HBM options.
4. Output results to `rdu_1000tops_sweep_results.csv`.
5. Generate a comprehensive analysis report in `RDU_1000TOPS_CO_DESIGN_REPORT.md`.
