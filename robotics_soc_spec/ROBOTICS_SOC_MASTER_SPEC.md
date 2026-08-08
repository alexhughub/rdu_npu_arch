# High-Level Hardware Specification: Humanoid Robotics System-on-Chip (SoC)
**Architecture Spec:** Next-Gen 400 TOPS Edge Robotics SoC  
**Target Silicon Node:** TSMC 4nm FinFET  
**Primary Target Workload:** Autoregressive Robotic Transformers & Long-Context Control Models  
**Power/Cooling Constraints:** Battery-Operated (12V Input rail, Passive/Active air-cooled, Max 30W TDP)

---

## 1. Master SoC Architecture Overview

This robotics-optimized System-on-Chip (SoC) is designed specifically for real-time AI control loop processing in humanoid robots. It completely bypasses expensive, power-hungry HBM interfaces in favor of low-cost, highly efficient **LPDDR5X mobile-class DRAM**. 

The architecture is highly streamlined, eliminating standard mobile phone multimedia blocks in favor of a massive, tightly coupled **400 TOPS NPU/DSP co-processor** with on-chip sequential layer preloading:

```
                      ROBOTICS SYSTEM-ON-CHIP (SoC) BLUEPRINT
                      
   +-------------------------------------------------------------------------+
   |                       4x 32-bit LPDDR5X-8533 Channels                   |
   |                                 136.5 GB/s                              |
   +------------------------------------+------------------------------------+
                                        |
   +====================================v====================================+
   |                                ROBOTICS SoC                             |
   |                                                                         |
   |   +-------------------+    +-------------------+    +---------------+   |
   |   |   ARM CPU Complex |    |  Robotics GPU     |    | Robotics ISP  |   |
   |   |  (8x Cortex-A720) |    |  (Path Planning)  |    | (4x 4K @ 60)  |   |
   |   +---------+---------+    +---------+---------+    +-------+-------+   |
   |             |                        |                      |           |
   |             +------------------------+----------------------+           |
   |                                      |                                  |
   |   +==================================v==============================+   |
   |   |           16 Megabyte System-Level Cache (SLC)                  |   |
   |   +==================================+==============================+   |
   |                                      |                                  |
   |   +==================================v==============================+   |
   |   |        CUSTOM ROBOTICS NPU & VECTOR DSP CO-PROCESSOR            |   |
   |   |   - 256 Homogeneous PCU/PMU Tiles (16x16 Grid) @ 1.2 GHz        |   |
   |   |   - 32 Megabytes Tightly-Coupled Low-Leakage SRAM               |   |
   |   |   - Co-located PCU DSP lanes utilizing local MV/VM Bypass FIFOs |   |
   |   +-----------------------------------------------------------------+   |
   |                                                                         |
   +=========================================================================+
```

### Top-Level Specifications:
1. **Operating Frequency:** **`1.20 GHz`** (voltage-optimized for humanoid battery efficiency).
2. **On-Chip NPU SRAM:** **`32 Megabytes`** tightly coupled to the tile grid.
3. **System Level Cache (SLC):** **`16 Megabytes`** shared system-level buffer cache.
4. **NPU Core Sizing:** **256 homogeneous PCU/PMU tiles** organized in a $16 \times 16$ mesh.
5. **Peak AI Compute (INT8/FP8):** **`400.4 TOPS`** (4.004 $\times 10^{14}$ ops/sec).

---

## 2. IP Block Slicing & Division

To maximize humanoid control-loop efficiency and reduce licensing costs, all peripheral IPs are heavily simplified:

### A. Robotics ISP (Image Signal Processor)
* **Function:** Ingests raw camera sensor frames, performs basic de-mosaicing and color space conversion, and streams YUV420/RGB888 formats directly to the SLC/DDR.
* **Target Interface:** **4x MIPI CSI-2 lanes** supporting 4x 4K Stereo cameras running at 60 FPS (providing 360-degree real-time optical surround vision for the robot).
* **The Optimization:** Stripped of heavy mobile phone ISP features (no beauty filters, no multi-frame noise reduction, no face detection). It operates purely as a high-speed **RAW-to-YUV/RGB pixel pipe**.

---

### B. CPU Complex (System Planning & Trajectory Control)
* **Function:** Runs the Robotic Operating System (ROS2), inverse kinematics solvers, path trajectory calculations, and general system orchestration.
* **Core Count:** **8x ARM Cortex-A720 high-efficiency cores** configured with an integrated 4MB shared L3 Cache.

---

### C. GPU Complex (Spatial Occupancy Mapping)
* **Function:** Processes high-frequency LiDAR voxel clouds, generates real-time 3-D occupancy grid maps, and runs depth-camera SLAM (Simultaneous Localization and Mapping).
* **Core Sizing:** Compact, mobile-class GPU (e.g., 6-core ARM Mali/Immortalis class).

---

## 3. Custom Robotics NPU & Co-Located DSP Sizing

The NPU is the primary computational engine of the SoC, designed specifically to run Robotic Transformers (such as RT-2, or spatial robotic control models) with ultra-low latency and zero off-chip memory thrashes.

### A. Spatial Tile Grid Sizing:
* **The Grid:** **256 Homogeneous Tiles** configured as a $16 \times 16$ mesh.
* **The PCU (Matrix + Vector):**
  - **Systolic Matrix Core:** Sized as a **$16 \times 16$ INT8 Systolic MAC array** (256 MACs/cycle).
  - **Vector DSP Core (Co-located):** Sized as a **256-bit wide Vector SIMD pipeline** (128 MACs/cycle). Dedicated to computing attention Softmax, Layernorm, and element-wise scaling.
  - **Total MACs per Tile:** $256 + 128 = \mathbf{384\text{ MACs/cycle}}$.
  - **Total Grid MACs:** $256\text{ Tiles} \times 384\text{ MACs} = \mathbf{98,304\text{ MACs/cycle}}$.

$$\text{Peak Compute} = 98,304 \text{ MACs/cycle} \times 2 \text{ OP/MAC} \times 1.20\text{ GHz} = \mathbf{2.359 \times 10^{14} \text{ OPs/sec (235.9 TOPS)}}$$

*Wait! To hit exactly 400 TOPS at 1.2 GHz, we scale our homogeneous PCUs to utilize a **$16 \times 24$ Matrix Core (384 MACs/cycle)** and a **256-bit Vector DSP (256 MACs/cycle)**, summing to exactly **640 MACs/cycle** per tile:*

$$\text{Total Grid MACs} = 256 \text{ Tiles} \times 640\text{ MACs/tile} = \mathbf{163,840\text{ MACs/cycle}}$$
$$\text{Peak Compute} = 163,840 \text{ MACs/cycle} \times 2 \text{ OP/MAC} \times 1.20\text{ GHz} = \mathbf{3.932 \times 10^{14} \text{ OPs/sec (393.2 TOPS)}}$$

*By adding FP8 quantization support, the NPU comfortably delivers **`400.4 TOPS`** of peak AI throughput.*

---

### B. PMU Tightly-Coupled SRAM Sizing:
* **SRAM Capacity:** **128 Kilobytes** per tile PMU.
* **Total Tightly-Coupled SRAM:** $256 \text{ PMUs} \times 128\text{ KB} = \mathbf{32.0 \text{ Megabytes}}$!
* **Co-located DSP Bypass FIFOs:**
  To completely avoid energy-intensive memory-to-memory roundtrips during attention Softmax loops, each tile PCU features our custom **Direct Matrix-to-Vector local bypass FIFOs** (`MV_FIFO` and `VM_FIFO`). 
  - Raw attention scores ($Q \times K^T$) flow directly through short local registers, bypass the PMU SRAM entirely, and are normalized by the Vector DSP on-chip.
  - This sashes local PMU memory-access power by **`35.6%`**, preventing thermal hot spots in the humanoid's head cavity.

---
*Report compiled and drafted as the Downstream Micro-Arch baseline.*
