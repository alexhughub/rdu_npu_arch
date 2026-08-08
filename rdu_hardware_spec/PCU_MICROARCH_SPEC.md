# Microarchitectural Specification: Programmable Compute Unit (PCU)
**Revision:** v1.0 (Downstream RTL Design Baseline)

This document defines the physical pipeline, sub-blocks, execution schemes, registers, boundary interfaces, area, power, and throughput parameters of the **Programmable Compute Unit (PCU)** inside each homogeneous tile of the RDU.

---

## 1. Top-Level Block Diagram & Interface Slices

The PCU is a software-configurable compute engine containing a high-throughput **Systolic GEMM Matrix Core** running in parallel with a **128-bit Vector SIMD Core**:

```
                       PCU MICROARCHITECTURAL BLOCK DIAGRAM
                       
                      +-----------------------------------+
                      |             PMU SRAM              |
                      +-----------------+-----------------+
                                        |
       +--------------------------------+--------------------------------+
       | Weight Port (256b)             | Activation Port (256b)         | Accumulate (512b)
       v                                v                                v
  +----+--------------------------------+--------------------------------+----+
  |                                 PCU TILE                                  |
  |                                                                           |
  |   +--------------------------+             +--------------------------+   |
  |   |    Matrix GEMM Core      |             |    Vector SIMD Core      |   |
  |   |    (16x24 PE MAC Array)  |             |     (128-bit VALU/SFU)   |   |
  |   +------------+-------------+             +------------+-------------+   |
  |                |                                        |                 |
  |                +--------------------+-------------------+                 |
  |                                     |                                     |
  |                                     v (Local Bus Bypass)                  |
  |                        +------------+-------------+                       |
  |                        |    PCU Local Register    |                       |
  |                        |     File (64x32b Regs)   |                       |
  |                        +------------+-------------+                       |
  |                                     |                                     |
  +-------------------------------------|-------------------------------------+
                                        v Writeback Bus (512b)
                              +---------+---------+
                              |     PMU SRAM      |
                              +-------------------+
```

### Physical Sizing Parameters (7nm Node):
* **Physical Area:** **`~0.22 mm²`** per PCU.
* **Active Power TDP:** **`~115 mW`** per PCU at 1.35 GHz.
* **Peak BF16 Compute:** **1.21 TFLOPS** per PCU.

---

## 2. PCU Sub-Block 1: Systolic GEMM Matrix Core

The Matrix Core is a hardwired systolic array optimized for dense matrix multiplication ($Y = W \times X$).

### A. Sub-Module Division:
1. **The PE MAC Grid:** Organized as a **$16 \times 24$ grid of Processing Elements (PEs)** (384 PEs total).
2. **PE Internals:** Each PE contains a **16-bit BF16/FP16 multiplier**, a **32-bit floating-point adder**, and a **32-bit local accumulator register** ($A_{\text{local}}$).
3. **Weight Latches:** A 16-bit local latch to pin model weight coefficients on-chip during weight-stationary execution.

```
                         PE CELL SCHEMATIC (MATRIX CORE)
                         
    Weight Port (W_in) ========> [ 16-bit Latch ]
                                        |
                                        v
    Act Port (X_in) ========> [ Multiplier (BF16) ] <=== (Weight Latch)
                                        |
                                        v  (32-bit Product)
    Accumulate (Y_in) =======> [ Floating-Point Adder ] <=== (A_local Accumulator)
                                        |
                                        v
                               [ 32-bit Accumulator ] ====> Output (Y_out)
```

### B. Micro-Arch Port Interface:
* **`PORT_W_IN` (256-bit):** Input weight channel from the PMU SRAM. Streams 16 BF16 weights per cycle.
* **`PORT_X_IN` (256-bit):** Input activation/query channel from the PMU SRAM. Streams 16 BF16 activations per cycle.
* **`PORT_Y_ACC` (512-bit):** Accumulator input channel from the PMU or adjacent PCU. Streams 16 32-bit FP32 accumulate frames.
* **`PORT_Y_OUT` (512-bit):** Compute output writeback channel to PMU. Streams 16 32-bit FP32 output frames.

---

## 3. PCU Sub-Block 2: Vector SIMD Core

The Vector Core executes non-GEMM layer arithmetic (Softmax, SwiGLU, LayerNorm) in parallel with GEMM, using a **128-bit wide Vector SIMD pipeline**.

### A. Pipeline Architecture:
The Vector Core operates on a **4-stage decoupled pipeline**:

```
                       VECTOR SIMD PIPELINE ARCHITECTURE
                       
   +--------------+     +--------------+     +--------------+     +--------------+
   |   Stage 1    |     |   Stage 2    |     |   Stage 3    |     |   Stage 4    |
   | Vector Fetch | ===>| Vector Decode| ===>|  VALU / SFU  | ===>| Vector Write |
   |  (REG/PMU)   |     | (Op Dispatch)|     | (ALU/Trans)  |     |   Back       |
   +--------------+     +--------------+     +--------------+     +--------------+
```

1. **VALU (Vector Arithmetic Logic Unit):** Contains 8 parallel 16-bit vector lanes. Executes vector addition, multiplication, and scaling on BF16/FP16 numbers.
2. **SFU (Special Function Unit):** Custom piece-wise linear approximation (LUT-based) pipeline. Executes transcendental operations: **exponential ($e^x$), inverse ($1/x$), and reciprocal square root ($1/\sqrt{x}$)** in a single cycle.

### B. Execution Schemes for Popular Algorithms:

#### 1. Attention Softmax:
Softmax is executed across three sequential SIMD loop steps utilizing the VALU and SFU:
* **Step 1 (Max Reduction):** VALU scans the active token activations to find the maximum element ($x_{\text{max}}$) to prevent exponential overflow.
* **Step 2 (Exponentiation):** SFU computes the exponential subtraction: $e^{x_i - x_{\text{max}}}$ using LUT lookup.
* **Step 3 (Normalization):** VALU sums the exponentials: $S = \sum e^{x_i - x_{\text{max}}}$. SFU computes $1/S$, and VALU multiplies all elements by the reciprocal to output normalized probabilities.

#### 2. SwiGLU Activation:
SwiGLU ($Swish(x \times W_{\text{gate}}) \times (x \times W_{\text{up}})$) is executed in a dual-path pipeline:
* **Path 1 (Swish):** VALU computes product $A = x \times W_{\text{gate}}$. SFU computes sigmoid $\sigma(A) = 1 / (1 + e^{-A})$. VALU computes Swish output $B = A \times \sigma(A)$.
* **Path 2 (Up-Projection):** VALU computes projection $C = x \times W_{\text{up}}$.
* **Path 3 (Merge):** VALU computes element-wise product $B \times C$ and writes back to PMU.

#### 3. Layer Normalization:
* VALU computes vector mean $\mu = \frac{1}{N}\sum x_i$ and variance $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$.
* SFU computes reciprocal square root: $1/\sqrt{\sigma^2 + \epsilon}$.
* VALU multiplies elements by the reciprocal, scales by $\gamma$, and shifts by $\beta$ to output normalized tokens.

#### 4. Element-Wise Addition & Scaling:
* VALU loads Vector $A$ from Port A and Vector $B$ from Port B, executes $A + B$ or $s \times A + B$ in 1 cycle, and writes back.

---

## 4. Register Files & Control Registers

Each PCU contains a dedicated register file to manage configuration states and loop counters:

### A. Local Register File:
* **PCU_GPR[0:63]:** 64 General-Purpose Registers (32-bit wide) to store scalar constants, loop variables, and address offsets.
* **PCU_VR[0:15]:** 16 Vector Registers (128-bit wide) to buffer intermediate vector states inside the VALU pipeline.

### B. PCU Configuration Registers:
* **`PCU_CTRL_REG` (32b):** Global control register.
  - Bit[0]: PCU Enable (1 = Active, 0 = Power gated).
  - Bit[2:1]: Execution Mode (00 = Weight Stationary, 01 = Input Stationary, 10 = Vector SIMD Solo).
  - Bit[4:3]: Data Precision (00 = BF16, 01 = FP16, 10 = FP8/INT8).
* **`PCU_LOOP_LIMIT` (32b):** Dedicated loop counter register. Triggers automatic hardware loop counters for GEMM steps without software branching overhead.

---

## 5. Popular Power-Saving Techniques Employed

To maximize energy efficiency inside the PCU, the following industry-popular techniques are implemented in RTL:

1. **Operand-Sensing Clock Gating:** 
   * Senses the inputs to the BF16 multipliers. If an input is exactly zero (null token/gated expert), the clock to the entire multiplier pipeline column is **gated (turned off)** on that cycle, saving up to **`35%`** of dynamic switching power!
2. **Sub-Threshold Leakage Control (Power Gating):**
   * PCU tiles are partitioned into 4 power domains. If a tile is unallocated by the compiler (e.g. during a small prefill phase), its power supply rail is physically isolated using on-chip header power switches, reducing idle leakage power to **`< 1%`**.

---
*Report compiled and structured as the Downstream Micro-Arch baseline.*
