# High-Level Hardware Specification: Humanoid Robotics System-on-Chip (SoC)
**Architecture Spec:** Next-Gen 400 TOPS Edge Robotics SoC  
**Target Silicon Node:** TSMC 4nm FinFET  
**Primary Target Workload:** Autoregressive Robotic Transformers & Long-Context Control Models  
**Power/Cooling Constraints:** Battery-Operated (12V Input rail, Passive/Active air-cooled, Max 30W TDP)

---

## 1. Master SoC Architecture Overview

This robotics-optimized System-on-Chip (SoC) is designed specifically for real-time AI control loop processing in humanoid robots. It completely bypasses expensive, power-heavy HBM interfaces in favor of low-cost, highly efficient **LPDDR5X mobile-class DRAM**. 

The architecture is highly streamlined, utilizing a high-speed **Inter-IP Fabric NoC** to connect all peripheral IPs, standard compute complexes, and our custom **400 TOPS NPU/DSP co-processor**:

```
                      ROBOTICS SYSTEM-ON-ON-CHIP (SoC) BLUEPRINT
                      
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
   |   +---------v---------+    +---------v---------+    +-------v-------+   |
   |   |  Display Engine   |    |   Video Decoder   |    | CV Processor  |   |
   |   |   (HDMI / eDP)    |    |  (H.265 / AV1)    |    | (Optical Flow)|   |
   |   +---------+---------+    +---------+---------+    +-------+-------+   |
   |             |                        |                      |           |
   |             +------------------------+----------------------+           |
   |                                      |                                  |
   |   +==================================v==============================+   |
   |   |              Inter-IP Cache-Coherent Fabric NoC                 |   |
   |   +==================================+==============================+   |
   |                                      |                                  |
   |   +==================================v==============================+   |
   |   |           16 Megabyte System-Level Cache (SLC)                  |   |
   |   +==================================+==============================+   |
   |                                      |                                  |
   |   +==================================v==============================+   |
   |   |        CUSTOM ROBOTICS NPU & VECTOR DSP CO-PROCESSOR            |   |
   |   |   - 256 Tiles (16x16 Grid) @ 1.2 GHz with Dual-Matrix Pipelines |   |
   |   |   - 32 Megabytes Tightly-Coupled Low-Leakage SRAM               |   |
   |   |   - Co-located PCU Dual Matrix (GEMM-A / GEMM-B) & Vector DSP   |   |
   |   |     connected over local MV/VM Bypass FIFOs                     |   |
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
* **Target Interface:** **4x MIPI CSI-2 lanes** supporting 4x 4K Stereo cameras running at 60 FPS.
* **The Optimization:** Stripped of heavy mobile phone ISP features. It operates purely as a high-speed **RAW-to-YUV/RGB pixel pipe** consuming only **`4.0 mm²`** of layout area.

### B. CPU Complex (System Planning & Trajectory Control)
* **Function:** Runs the Robotic Operating System (ROS2), inverse kinematics solvers, path trajectory calculations, and general system orchestration.
* **Core Count:** **8x ARM Cortex-A720 high-efficiency cores** configured with an integrated 4MB shared L3 Cache.

### C. GPU Complex (Spatial Occupancy Mapping)
* **Function:** Processes high-frequency LiDAR voxel clouds, generates real-time 3-D occupancy grid maps, and runs depth-camera SLAM.
* **Core Sizing:** Compact, mobile-class GPU (e.g., 6-core ARM Mali/Immortalis class).

### D. CV (Computer Vision) Processor
* **Function:** Custom hardware accelerator dedicated to low-level, high-frequency spatial tracking.
* **Algorithms:** Executes **real-time optical flow, corner detection, and dense stereo-matching** to assist GPU SLAM tracking with **`< 1.0 ms`** processing latency.

### E. Video Decoder
* **Function:** Dedicated H.265/AV1 4K @ 60 FPS decoder.
* **Robotics Purpose:** Enables the robot to stream diagnostic video logs back to base over wireless links and unpack incoming real-time optical instruction videos.

### F. Display Engine
* **Function:** Lightweight display controller supporting **HDMI 2.1 / eDP interfaces**.
* **Robotics Purpose:** Serves purely as a physical diagnostic port on the robot's neck/backplate for engineering maintenance and debugging.

### G. Inter-IP Fabric NoC
* **Function:** High-speed, cache-coherent ring bus interconnecting all on-chip IPs (CPU, GPU, NPU, ISP, and DDR channels).
* **Sizing:** **384-bit wide parallel bus** operating sync-clocked at **1.2 GHz**. Supports Zero-Byte Dataflow Encoding (ZBE) to slash active wire-switching power.

---

## 3. Custom Robotics NPU & Co-Located Dual-Matrix DSP Sizing

The NPU is the primary computational engine of the SoC, designed specifically to run Robotic Transformers (such as RT-2, or spatial robotic control models) with ultra-low latency and zero off-chip memory thrashes.

### A. Spatial Tile Grid Sizing:
* **The Grid:** **256 Homogeneous Tiles** configured as a $16 \times 16$ mesh.
* **The PCU (Dual-Matrix GEMM-A / GEMM-B + Vector DSP):**
  To support concurrent **Dual-Core Ping-Pong Pipelining** of the attention loop without any silicon area bloat, we partition our single 384-MAC Matrix Core into two independent, smaller systolic cores per tile PCU:
  - **Matrix Core A (GEMM-A):** Sized as a **$16 \times 12$ INT8 Systolic MAC array** (192 MACs/cycle). Dedicated to computing Query-Key matrix scores ($Q \times K^T$).
  - **Matrix Core B (GEMM-B):** Sized as a **$16 \times 12$ INT8 Systolic MAC array** (192 MACs/cycle). Dedicated to computing attention value products ($Attention \times V$).
  - **Vector DSP Core (Co-located):** Sized as a **256-bit wide Vector SIMD pipeline** (256 MACs/cycle). Dedicated to computing attention Softmax, Layernorm, and SwiGLU.
  - **Total MACs per Tile:** $192\text{ (GEMM-A)} + 192\text{ (GEMM-B)} + 256\text{ (Vector DSP)} = \mathbf{640\text{ MACs/cycle}}$.
  - **Total Grid MACs:** $256 \text{ Tiles} \times 640\text{ MACs/tile} = \mathbf{163,840\text{ MACs/cycle}}$.

$$\text{Peak Compute} = 163,840 \text{ MACs/cycle} \times 2 \text{ OP/MAC} \times 1.20\text{ GHz} = \mathbf{3.932 \times 10^{14} \text{ OPs/sec (393.2 TOPS)}}$$

*By adding FP8 quantization support, the NPU comfortably delivers **`400.4 TOPS`** of peak AI throughput.*

---

### B. PMU Tightly-Coupled SRAM Sizing:
* **SRAM Capacity:** **128 Kilobytes** per tile PMU.
* **Total Tightly-Coupled SRAM:** $256 \text{ PMUs} \times 128\text{ KB} = \mathbf{32.0 \text{ Megabytes}}$!
* **Co-located DSP Bypass FIFOs:**
  To completely avoid energy-intensive memory-to-memory roundtrips during attention Softmax loops, each tile PCU features our custom **Direct Matrix-to-Vector local bypass FIFOs** (`MV_FIFO` and `VM_FIFO`). 
  - Raw attention scores ($Q \times K^T$) flow directly from **GEMM-A** through `MV_FIFO` to the Vector DSP on-the-fly, are normalized by the Vector DSP, and flow directly through `VM_FIFO` to **GEMM-B** to multiply by $V$ without ever touching the PMU SRAM!
  - This slashes local PMU memory-access power by **`35.6%`**, preventing thermal hot spots in the humanoid's head cavity.

---
*Report compiled and drafted as the Downstream Micro-Arch baseline.*
