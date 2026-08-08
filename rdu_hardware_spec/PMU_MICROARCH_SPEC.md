# Microarchitectural Specification: Programmable Memory Unit (PMU)
**Revision:** v1.0 (Downstream RTL Design Baseline)

This document defines the physical memory organization, bank partitioning, Address Generation Unit (AGU), local bus interfaces, area, power, and bandwidth parameters of the **Programmable Memory Unit (PMU)** inside each homogeneous tile of the RDU.

---

## 1. Top-Level Block Diagram & Interface Slices

Each PMU contains a **128 Kilobytes** high-speed dual-ported SRAM cache. To completely bypass bank access collisions, the memory features a multi-bank partitioned layout managed by a **16x16 Crossbar Matrix Switch**:

```
                       PMU MICROARCHITECTURAL BLOCK DIAGRAM
                       
                                +-------------------+
                                |    NoC Router     |
                                +---------+---------+
                                          |
                        +-----------------+-----------------+
                        | NoC Read (256b)                   | NoC Write (256b)
                        v                                   v
             +----------+-----------------------------------+----------+
             |                         PMU TILE                        |
             |                                                         |
             |   +-------------------------------------------------+   |
             |   |            16x16 Crossbar Matrix Switch         |   |
             |   +--------+---------+-----------+---------+--------+   |
             |            |         |           |         |            |
             |            v         v           v         v            |
             |        +-------+ +-------+   +-------+ +-------+        |
             |        | Bank0 | | Bank1 |...| Bank14| | Bank15|        |
             |        | (8KB) | | (8KB) |   | (8KB) | | (8KB) |        |
             |        +-------+ +-------+   +-------+ +-------+        |
             |            |         |           |         |            |
             |            +---------+-----------+---------+            |
             |                               |                         |
             |                               v                         |
             |                  +------------+-------------+           |
             |                  |  Address Generation Unit |           |
             |                  |  (AGU & INT4 Quantizer)  |           |
             |                  +------------+-------------+           |
             |                               |                         |
             +-------------------------------|-------------------------+
                                             |
                  +--------------------------+--------------------------+
                  | PCU Read (512b)                                     | PCU Write (512b)
                  v                                                     v
         +--------+--------+                                   +--------+--------+
         |   PCU Matrix    |                                   |    PCU Vector   |
         |   GEMM Core     |                                   |    SIMD Core    |
         +-----------------+                                   +-----------------+
```

### Physical Sizing Parameters (7nm Node):
* **Physical Area:** **`~0.05 mm²`** per PMU (including SRAM core and AGU).
* **Active Power TDP:** **`~45 mW`** per PMU at 1.35 GHz.
* **Aggregate Bandwidth:** **172.8 Gigabytes/sec** local read/write bandwidth.

---

## 2. PMU SRAM Core & Multi-Bank Crossbar Slicing

To support simultaneous, multi-port local compute and NoC routing without stalls:

### A. Memory Slicing:
* **SRAM Capacity:** **128 Kilobytes** (Aggregate Grid SRAM = **`128 Megabytes`**).
* **Memory Banking:** divided into **16 independent memory banks (8KB per bank)**.
* **SRAM Cell Layout:** Built using **8T dual-ported SRAM cells**. Port A is dedicated for local reads; Port B is dedicated for concurrent local/NoC writes.

### B. The 16x16 Crossbar Switch:
* Implements a **16x16 non-blocking crossbar routing matrix**.
* Allows up to **16 parallel read/write memory operations on a single clock cycle**, as long as the accesses target different physical banks.
* **Collision Penalty:** If the PCU vector writeback and the NoC weight prefetch target the same 8KB bank on the same cycle, the PMU arbiter injects a **1-cycle hardware stall** on the NoC write port, prioritizing compute writeback.

---

## 3. PMU Sub-Block 2: Address Generation Unit (AGU) & INT4 Quantizer

The AGU manages physical memory indexing and hardware-accelerated integer compression.

### A. Dynamic Address Generation:
* Implements 4 independent **strided address generators** to support complex 2-D matrix loops.
* **Tiled Transpose:** Supports 1-cycle physical matrix transpose address generation (allowing the PCU to read a matrix column-wise instead of row-wise on the fly, with zero memory copy overhead).

### B. INT4 Hardware Quantization Engine:
* Contains dedicated, low-latency DSP pipelines to convert floating-point activations to quantized integers.
* **Compression Pathway (Write):** Senses incoming FP16/BF16 data, computes scale factors, quantizes to **INT4 (4-bit)**, and packs 4 elements into a single 16-bit SRAM word.
* **Decompression Pathway (Read):** Unpacks the 16-bit word, multiplies by local scale factor, and restores FP16 vectors to feed the PCU ALUs.
* **The Sizing Victory:** Quantizing intermediate activations slashes the active KV-cache on-chip footprint from 500 MB down to **15.6 MB per layer**, allowing a **400,000-token sequence context** to fit completely on-chip inside the PMUs with **zero DRAM spills**.

---

## 4. Boundary Bus Interfaces

The PMU acts as the local routing junction inside the tile, communicating with the PCU and the NoC over three dedicated, parallel buses:

1. **PCU Read Bus (512-bit):**
   * High-speed, local bus from PMU banks to PCU input buffers.
   * Delivers **16 FP32 or 32 BF16/FP16 elements per cycle** (86.4 GB/s local read bandwidth).
2. **PCU Write Bus (512-bit):**
   * Writeback bus from the PCU vector core to PMU banks.
   * Feeds **16 FP32 vector products per cycle** into the PMU.
3. **NoC Boundary Bus (256-bit):**
   * Interconnect bus from the PMU to the local NoC Router.
   * Streams **256-bit wide packetized flits per cycle** (43.2 GB/s link speed) to route data to adjacent tiles or HBM interfaces.

---

## 5. Register Files & Control Registers

Each PMU contains configuration registers managed by the tile controller:

* **`PMU_CTRL_REG` (32b):** Global memory control register.
  - Bit[3:0]: Memory Partition Mode (e.g. 0001 = 32KB Weights, 64KB Activations, 32KB KV-cache; 0010 = 64KB KV-cache, 32KB Weights, 32KB Activations).
  - Bit[4]: AGU Quantization Enable (1 = INT4 compression active, 0 = Raw FP16 bypass).
* **`PMU_BANK_STATUS` (16b):** Dynamic 1-bit status flag per bank. Indicates whether a bank is locked by an active compute pipeline or available for NoC prefetch.
* **`AGU_STRIDE_REG` (32b):** Configures address generation stride intervals for 2-D matrix streaming loops.

---

## 6. Popular Power-Saving Techniques Employed

To minimize SRAM active leakage and switching power, the PMU implements:

1. **Bitline Charging Gating:**
   * Senses the write mask. If only half of the 512-bit bus contains active write data (e.g. during an unaligned vector write), the clock to the write-drivers of the empty banks is **gated**, saving **`45%`** of write-switching power.
2. **SRAM Sleep Mode (Retention Gating):**
   * If a tile's memory region is unallocated during a model phase, the PMU switches to **Deep Sleep Mode**, lowering the SRAM core supply voltage to **`0.45V`** (retention margin limit). This slashes active leakage power by **`70%`** while safely preserving stored state.

---
*Report compiled and structured as the Downstream Micro-Arch baseline.*
