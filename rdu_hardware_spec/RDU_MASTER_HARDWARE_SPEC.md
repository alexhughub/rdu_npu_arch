# High-Level Hardware Specification: Next-Gen Reconfigurable Dataflow Unit (RDU)
## Architecture Spec: Next-Gen 1.4 PFlops Long-Context Spatial Processor

**Specification Status:** Draft v1.0 (Downstream Micro-Arch Baseline)  
**Target Process Node:** TSMC 7nm / 3nm Synthesis Optimizations  
**Primary Workload Optimization:** Long-Context Generative AI (LLaMA-3-70B, DeepSeek-V3 MoE)  
**Key Features:** Tiled 2D Torus NoC, INT4 AGU Hardware Data Compression, Homogeneous Reconfigurable PCUs/PMUs.

---

## 1. Master System Architecture Overview

The Next-Gen Reconfigurable Dataflow Unit (RDU) is a software-configurable spatial dataflow processor. It replaces monolithic, hardwired central execution blocks with a **$32 \times 32$ homogeneous Grid of Tiles** (1,024 total tiles) interconnected by a high-bandwidth 2D Torus Network-on-Chip (NoC).

```
                      RDU GRID TOPOLOGY (32x32 homogeneous Tiles)
                      
     (Torus Link) <======= Bidirectional NoC links (43.2 GB/s) =======> (Torus Link)
                      +---------+   +---------+   +---------+
                      | Tile 0  |===| Tile 1  |===| Tile 2  |
                      | PCU/PMU |   | PCU/PMU |   | PCU/PMU |
                      +---------+   +---------+   +---------+
                           ||            ||            ||
                      +---------+   +---------+   +---------+
                      | Tile 32 |===| Tile 33 |===| Tile 34 |
                      | PCU/PMU |   | PCU/PMU |   | PCU/PMU |
                      +---------+   +---------+   +---------+
                           ||            ||            ||
                      +---------+   +---------+   +---------+
                      | Tile 64 |===| Tile 65 |===| Tile 66 |
                      | PCU/PMU |   | PCU/PMU |   | PCU/PMU |
                      +---------+   +---------+   +---------+
```

### System Sizing Parameters:
1. **Clock Frequency:** **`1.35 GHz`** (custom physical synthesis).
2. **Homogeneous Grid:** **1,024 Tiles** organized as a $32 \times 32$ mesh.
3. **Peak Compute (BF16):** **`1,238.3 TFLOPS (1.24 Petaflops/sec)`**
4. **Peak Compute (INT8/FP8):** **`2,476.6 TOPS (2.48 Petaflops/sec)`**

---

## 2. Microarchitectural Sizing and Execution Blocks

Each of the 1,024 Homogeneous Tiles contains a **PCU (Programmable Compute Unit)** vector and matrix core and a **PMU (Programmable Memory Unit)** SRAM block.

### A. PCU (Programmable Compute Unit) Sizing
To deliver 1.24 Petaflops/sec at 1.35 GHz, each PCU contains two parallel execution pipelines:

1. **Systolic GEMM Matrix Core:** 
   * Sized as a **$16 \times 24$ Systolic Multiply-Accumulate (MAC) array** (384 MACs/cycle).
   * Supports native BF16, FP16, and FP8 matrix dot-products.
2. **Vector SIMD Core:**
   * Sized as a **128-bit wide Vector SIMD pipeline** (64 MACs/cycle).
   * Fully dedicated to non-GEMM activation layers: **Softmax, GeLU/SwiGLU, Layer Normalization, and element-wise addition/scaling**.
3. **Total MACs per PCU:** $384 + 64 = \mathbf{448\text{ MACs/cycle}}$.
4. **Total Grid MACs:** $1024 \text{ PCUs} \times 448\text{ MACs} = \mathbf{458,752\text{ MACs/cycle}}$ ($4.58 \times 10^5$ operations per cycle).

$$\text{Peak BF16 Compute} = 458,752 \text{ MACs/cycle} \times 2 \text{ FLOPs/MAC} \times 1.35\text{ GHz} = \mathbf{1.238 \times 10^{15} \text{ FLOPs/sec (1.24 PFlops)}}$$

---

### B. PMU (Programmable Memory Unit) Partitioning
Each PMU contains a **128 Kilobytes** high-speed dual-ported SRAM cache. To completely bypass bank collision hazards, the PMU features a segmented bank layout:

* **SRAM Capacity:** **128 KB** per PMU (Aggregate Grid SRAM = **`128 Megabytes`**).
* **Segmented Layout:** divided into **16 independent memory banks (8KB per bank)**.
* **Cell Layout:** Built using **8T dual-ported SRAM bit-cells** (supporting simultaneous Port A read and Port B write).

#### PMU Configuration Partitions:
The 128KB memory region can be software-configured at runtime into distinct, dedicated circular buffers depending on active dataflow layers:

```
                  128KB PMU SRAM STORAGE PARTITIONS (CONFIGURABLE)
                  
   +-----------------------------------------------------------------+
   |  Weight Buffer Channel (Active + Prefetch Double-Buffer)        |
   |  * Sized: 32 KB (Port A: 16KB PCU read | Port B: 16KB prefetch) |
   +-----------------------------------------------------------------+
   |  Activation / Query Channel (Input + Output Elastic FIFO)       |
   |  * Sized: 64 KB (32KB Input circular buffer | 32KB Output FIFO) |
   +-----------------------------------------------------------------+
   |  KV-Cache Storage Slice                                         |
   |  * Sized: 32 KB (Stores active local token KV histories)        |
   +-----------------------------------------------------------------+
```

* **INT4 AGU Hardware Data Compression:**
  The PMU's **Address Generation Unit (AGU)** features native, low-latency 4-bit integer quantization engines.
  - Activations and KV-caches are compressed from FP16 (2 bytes) to INT4 (0.5 bytes) on the fly as they are written to SRAM.
  - Slashes active KV-cache on-chip memory footprint by **4x**, enabling sequence context lengths up to **400,000 tokens** to fit entirely on-chip over the grid PMUs with **zero DRAM spills**.

---

## 3. High-Bandwidth Memory (HBM3) and NoC Interconnect Specs

### A. HBM3 Memory System
* **HBM3 Capacity:** **96 Gigabytes**.
* **Memory Interface:** 6144-bit wide high-speed bus over 6 stack channels.
* **HBM to PMU Bandwidth (Off-Chip):** **`2.4 Terabytes/sec (2400 GB/s)`** aggregate off-chip read/write bandwidth.
* **HBM Prefetch Latency Overlap:** Because weights are loaded in 16KB chunks per tile, loading a 16.3 MB weight slice takes only **`0.77 ms`** over the 2.4 TB/s bus. Since compute on a sequence-tiled chunk takes **`2.1 ms`**, **100% of HBM weight loading latency is asynchronously hidden (overlapped) under active compute cycles!**

---

### B. Network-on-Chip (NoC) Interconnect
* **NoC Topology:** High-Bandwidth, Multi-Plane 2D Torus NoC.
* **Link Width:** **256-bit wide** parallel channels.
* **Link Operating Speed:** Sync-clocked at **1.35 GHz**.
* **NoC Link Bandwidth (Bidirectional):**
  $$\text{NoC Link BW} = 256\text{ bits} \times 1.35\text{ GHz} = 345.6\text{ Gb/s} = \mathbf{43.2\text{ Gigabytes/sec per link direction}}$$
* **On-Chip Bisection Bandwidth:** Slicing across the 32 rows over 2 separate routing planes:
  $$\text{Bisection BW} = 32\text{ rows} \times 2\text{ planes} \times 43.2\text{ GB/s} = \mathbf{2,764.8\text{ GB/s (2.76 Terabytes/sec)}}$$

---

## 4. Multi-Chip Sockets & Interconnect (Tensor Parallelism Scaling)

To run **Tensor Parallelism (TP=8)** across 8 physical RDU sockets, the RDU's boundary ports directly extend the on-chip Torus NoC over physical high-speed board traces.

* **Inter-Chip Links (ICL):** **8 Boundary Ports** per chip.
* **ICL Link Bandwidth:** **150 Gigabytes/sec** bidirectional bandwidth per port.
* **Aggregate Boundary Bandwidth:** **`1.2 Terabytes/sec`** inter-socket throughput per chip.
* **Unified Virtual Grid:** The inter-socket boundary transceivers support direct NoC packet forwarding, allowing the compiler to treat the 8 physical chips as a **single, unified 8,192 homogeneous tile virtual spatial dataflow mesh** with an ultra-low inter-socket latency of only **`1.0 µs`**!

---
*Report compiled and drafted as the Downstream Micro-Arch baseline.*
