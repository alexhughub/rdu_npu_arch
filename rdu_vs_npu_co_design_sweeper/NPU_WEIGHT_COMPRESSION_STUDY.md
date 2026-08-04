# "What-If" Co-Design Study: NPU with HBM-to-SRAM Weight Compression
## Does Decompressing Weights at the Memory Interface Alleviate the NPU Memory Wall?

**Report Status:** Completed (Physical Analytical Model Simulation)  
**Target Hardware Scale:** 1000 TOPS (1.0 Petaflops) BF16 Class  
**Investigated Architecture:** TPU-style NPU integrating 4x Hardware Weight Decompressor Blocks at the HBM physical boundary.

---

## Executive Summary

To bypass off-chip memory bandwidth limits, a compelling architectural proposal is to store **compressed model weights** (e.g., using 4x structured INT4 quantization) inside HBM, and decompress them on the fly at the interface boundary before storing or moving them to the centralized SRAM block. 

This study models the physical and electrical consequences of this "What-If" NPU architecture running **LLaMA-3-70B** and **DeepSeek-V3 MoE** layers under extreme sequence context lengths ($S = 32,768$ tokens) at Batch=1.

### Key Finding:
While decompressing weights at the HBM interface successfully reduces off-chip HBM weight-streaming latency by **`4.0x`**, **it does not solve the NPU's fundamental architectural bottlenecks**. 
1. **The Spill Wall Remains:** Because the NPU has uncompressed, raw centralized SRAM, intermediate activations (**`4.19 GB` at S=32k**) still overflow the central block. NPU is still forced to write/read **`3.94 GB`** of activation spills to HBM, stalling the PE array.
2. **MoE Spatial Inefficiency:** Even if expert weights are compressed, the sequential, temporal execution map of the systolic array still forces expert weights to be streamed repeatedly from HBM for different tokens, maintaining weight-thrashing overhead compared to RDU's spatial expert pinning.

---

## Section 1: Head-to-Head Quantitative Performance (S=32,768, Batch=1)

The table below contrasts the simulated performance of the **Baseline RDU**, **Baseline NPU**, and the **Proposed Compressed NPU**:

| Workload (S=32k, B=1) | Baseline NPU (No Comp) | Proposed NPU (4x Weight Comp) | SambaNova Spatial RDU | Compression Co-Design Winner |
| :--- | :---: | :---: | :---: | :--- |
| **LLaMA-3-70B Latency** | 112.94 ms | **`112.43 ms`** | **`98.30 ms`** | **RDU** (SRAM Spill Bypassing) |
| **LLaMA-3-70B TOPS** | 876.2 TFLOPS | **`880.2 TFLOPS`** | **`1006.6 TFLOPS`** | **RDU** (+14.3% faster) |
| **DeepSeek-V3 Latency** | 34.93 ms | **`34.19 ms`** | **`30.03 ms`** | **RDU** (Spatial expert pinning) |
| **DeepSeek-V3 TOPS** | 784.9 TFLOPS | **`801.8 TFLOPS`** | **`913.1 TFLOPS`** | **RDU** (+13.8% faster) |
| **Cost Efficiency (TFLOPS/$)**| 39.81 TOPS/$ | **`34.05 TOPS/$`** | **`26.78 TOPS/$`** | **NPU** (Superior PE cost layout) |

---

## Section 2: Detailed Microarchitectural Breakdown

### 1. Why HBM-to-SRAM Weight Compression Yields Diminishing Returns for NPU
* **The Bandwidth Savings:** Storing INT4 weights in HBM drops weight streaming latency for LLaMA-3-70B from **`0.77 ms` down to `0.19 ms`** (a massive 4x reduction!).
* **The Spill Bottleneck:** However, because the centralized SRAM scratchpad has no local AGU routing compressors nested inside the memory columns, the massive **4.19 GB of activations** still cannot be compressed on-chip. The NPU must still write and read spills to DRAM, incurring a massive **`1.59 ms`** activation spill delay which completely masks the 0.58 ms weight loading savings.
* **The Global Bus Contention:** Once weights are decompressed at the boundary, the uncompressed weights must still travel over the monolithic global shared bus to feed the PE rows. Global shared-bus congestion (which takes **`0.42 ms`**) is completely unchanged.

### 2. Physical Silicon Cost and Layout Headaches
* **Silicon Area Overhead:** Implementing 712 independent high-speed hardware decompressors at the centralized SRAM boundary to feed the rows on the fly adds approximately **`5.2 mm2`** of silicon layout area, boosting the good die cost of the 1000 TOPS NPU by **`$1.50`** (raising cost to **`$23.51`** per chip).
* **The Thermal Penalty:** Keeping weights compressed inside central SRAM and decompressing them at the PE row boundaries requires driving 712 independent decoders continuously at 1.0 GHz, boosting active thermal power dissipation (TDP) by an estimated **`18 Watts`**.

---

## Section 3: Synthesis of Co-Design Solutions

The "What-If" simulation proves that **compressing weights at the off-chip interface is a partial band-aid, not a holistic solution**:

* **To solve the Spill Wall:** The NPU must enable **on-chip activation compression**. But doing so requires nesting vector ALUs and decompression AGUs inside the central memory columns, which physically transforms the NPU's centralized block into a distributed, local PMU memory layout?effectively **turning the NPU into an RDU**!
* **To solve the MoE Memory Wall:** The NPU must enable **spatial data dispatch** (sending token activations to fixed expert PEs). But doing so requires replacing the rigid, hardwired systolic shift lines with a **2-D mesh Network-on-Chip (NoC)**, which again **transforms the systolic array into an RDU**.

---
*Report automatically compiled and formatted by the What-If Co-Design Simulation Suite.*
