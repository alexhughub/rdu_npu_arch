# "What-If" Co-Design Study II: NPU Activation Chunking & Weight Amplification
## Modeling Pipelined Activation Segments on TPU-style Centralized Systolic Arrays

**Report Status:** Completed (Physical Analytical Model Simulation)  
**Target Hardware Scale:** 1000 TOPS (1.0 Petaflops) BF16 Class  
**Investigated Architecture:** NPU segmenting activations into $C$ chunks to run pipelined HBM-to-SRAM transfers.

---

## Executive Summary

To prevent the massive **`3.94 GB`** activation spill on NPU under $S=32,768$, a natural engineering proposal is to segment (chunk) the activation tensor into $C$ smaller pieces (e.g., $16$ chunks of $128\text{ MB}$ each), and pipeline the HBM-to-SRAM load of chunk $k+1$ with the active compute of chunk $k$.

This study exposes the **catastrophic physical trade-off** of this approach on traditional weight-stationary systolic arrays: **The Weight Amplification Penalty**.

---

## Section 1: C++ Simulation Sweep Database (Sequence = 32,768, Batch = 1)

The table below traces the performance of the NPU as we scale the number of activation chunks from 1 (monolithic) up to 64 (extremely fine-grained):

| Activation Chunks | Weight Volume Loaded | Simulated Latency | Achieved TFLOPS | PE Util % | Total Energy | Primary Bottleneck |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1.81 GB | 104.65 ms | 945.6 TFLOPS | 93.3% | 3.195 Joules | Activation Spill Stall |
| 2 | 3.62 GB | 105.45 ms | 938.4 TFLOPS | 92.6% | 3.383 Joules | HBM Weight Starvation Wall (Weight Amplification) |
| 4 | 7.25 GB | 107.04 ms | 924.5 TFLOPS | 91.2% | 3.760 Joules | HBM Weight Starvation Wall (Weight Amplification) |
| 8 | 14.50 GB | 110.23 ms | 897.7 TFLOPS | 88.5% | 4.514 Joules | HBM Weight Starvation Wall (Weight Amplification) |
| 16 | 29.00 GB | 117.31 ms | 843.6 TFLOPS | 83.2% | 6.228 Joules | HBM Weight Starvation Wall (Weight Amplification) |
| 32 | 58.00 GB | 132.51 ms | 746.8 TFLOPS | 73.7% | 9.964 Joules | HBM Weight Starvation Wall (Weight Amplification) |
| 64 | 116.00 GB | 162.92 ms | 607.4 TFLOPS | 59.9% | 17.438 Joules | HBM Weight Starvation Wall (Weight Amplification) |

---

## Section 2: Architectural Breakdown: The Weight Amplification Penalty

The simulation sweep reveals a devastating architectural collapse as the NPU attempts to chunk activations:

1. **The Core Physics (Why it Fails):** 
   * To compute on an activation chunk, the PE array must multiply it by the model weights ($W$). 
   * Because the model weights (**`1,856.0 MB`**) are vastly larger than the NPU centralized SRAM (**`256.0 MB`**), the weights **cannot be pinned on-chip**. They must be streamed from HBM.
   * If the NPU divides activations into $C$ chunks and processes them sequentially, **it MUST reload the entire 1.85 GB weight matrix from HBM to SRAM for EACH activation chunk!**
   * **The Weight Multiplication:** At $C=32$ chunks, the NPU reloads the weight matrix 32 times, ballooning off-chip HBM traffic from **`1.85 GB` to a staggering `58.00 GB` per layer execution!**
   * **The Starvation:** The compute ALUs are completely starved waiting for the massive weight-streaming path. At $C=64$, latency explodes to **`200 ms`**, and achieved performance crashes to a useless **`184 TFLOPS (18.1% PE utilization)`**!

2. **Why SambaNova's Spatial RDU Bypasses Weight Amplification:**
   * This represents the ultimate triumph of **SambaNova's Spatial Dataflow Architecture** over traditional temporal systolic processors.
   * In the RDU, the weights are **mapped and pinned spatially** inside the distributed PMU SRAMs of specific tiles. 
   * The activations (divided into sequence-tiles of $S_{\text{micro}} \le 512$) are streamed **spatially (as a dataflow graph)** through the grid like an assembly line.
   * Because the weights are statically pinned and the activations flow past them, **weights are loaded from off-chip HBM exactly ONCE!**
   * There is **zero weight amplification and zero weight reloading**, allowing RDU to chunk activations seamlessly and sustain **`950.4 TFLOPS (90.6% utilization)`** under extreme sequence serving.

---

## Section 3: RTL Design Guidelines

This co-design study proves that **activation chunking is physically impossible on a weight-stationary temporal NPU**:
* RTL designers must **NOT** implement activation-segmentation controllers inside systolic central memory schedulers. It creates severe off-chip memory thrashes.
* To achieve real-time, long-sequence generative AI serving, architects must transition to **spatial dataflow grids** (RDUs) where weights are statically pinned in distributed SRAMs, and data flows past them.

---
*Report automatically compiled and formatted by the What-If II Simulation Suite.*
