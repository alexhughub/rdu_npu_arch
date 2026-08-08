# Memory Bandwidth & Capacity Sizing Analysis: Robotics SoC

This document evaluates the physical memory bandwidth (DDR) and capacity requirements of our humanoid robotics SoC under peak concurrent execution loops.

---

## Section 1: Memory Interface Specification

To maintain a low power profile suitable for battery-operated humanoid robots, our SoC utilizes mobile-class **LPDDR5X DRAM** instead of expensive, thermal-heavy HBM systems:

* **Memory Standard:** LPDDR5X-8533
* **Channel Configuration:** **4x 32-bit physical channels** (128-bit wide aggregate bus).
* **Bus Speed:** **`8,533 MT/s`** (Megatransfers per second).
* **Peak Aggregate DDR Bandwidth:**
  $$\text{Peak DDR BW} = \frac{8,533\text{ MT/s} \times 128\text{ bits}}{8 \text{ bits/byte}} = \mathbf{136,528 \text{ MB/s} \approx 136.5\text{ Gigabytes/sec}}$$

---

## Section 2: Memory Bandwidth Slicing per IP Block

The table below outlines the concurrent memory bandwidth (DDR) consumed by each individual IP block under active, real-world robotic control loops:

| IP Block | Primary Memory Transaction | Data Format | Active DDR Read/Write | DDR BW Duty Cycle |
| :--- | :--- | :---: | :---: | :---: |
| **Robotics ISP** | 4x 4K Stereo Camera feeds @ 60 FPS | RGB888 / YUV420 | **`5.97 GB/s`** (Writes) | 4.4% |
| **CPU Complex** | ROS2 planners, IK Solvers, Trajectories| FP32 / Scalars | **`15.00 GB/s`** (R/W) | 11.0% |
| **Robotics GPU** | LiDAR Voxel maps & 3-D Occupancy SLAM | FP32 Vectors | **`25.00 GB/s`** (R/W) | 18.3% |
| **NPU Weight Stream** | Autoregressive Layer-by-Layer Preload | FP8 Weights | **`80.00 GB/s`** (Reads) | 58.6% |
| **NPU Activations** | Tightly coupled S-tiling loops | INT4 (PMU Cached) | **`0.00 GB/s`** (On-Chip) | **0.0% (Zero-Spill)** |
| **System Overhead** | Display output, PCIe peripheral buffer | RGB / Controls | **`3.00 GB/s`** | 2.2% |
| **Total Co-execution**| **All IP blocks active concurrently** | ? | **`128.97 GB/s`** | **94.5% (Safe margin)**|

---

## Section 3: Deep Microarchitectural BW Sizing Calculations

```
                     CONCURRENT DDR BANDWIDTH ALLOCATION
                     
   LPDDR5X-8533 Interface Bus (136.5 GB/s Peak)
   |
   +===[  58.6% ]===> NPU Layer Weight Streaming (80.0 GB/s)
   |
   +===[  18.3% ]===> GPU Spatial Occupancy mapping (25.0 GB/s)
   |
   +===[  11.0% ]===> CPU Navigation & ROS2 Planners (15.0 GB/s)
   |
   +===[   4.4% ]===> ISP Camera Ingestion Writes (5.97 GB/s)
   |
   +===[   2.2% ]===> Telemetry & Displays (3.0 GB/s)
   |
   +===[   5.5% ]===> Unallocated DDR Safety Headroom (7.53 GB/s)
```

### 1. Robotics ISP Camera Ingestion (DDR Write Sizing)
* **The Ingest:** The robot has 4x 4K stereo cameras capturing surround environments for 360-degree safety and navigation.
* **RAW Sensor Input:** 4K 10-bit at 60 FPS streams $3840 \times 2160 \times 10 \times 60 = 4.97\text{ Gbps} = 622\text{ MB/s}$ per camera. Slicing 4 cameras = **`2.49 GB/s`** into the ISP.
* **YUV/RGB Convert Write:** The ISP converts RAW pixels to uncompressed RGB888 for the vision transformers:
  $$\text{RGB888 4K @ 60 FPS} = 3840 \times 2160 \times 3\text{ Bytes} \times 60 \text{ frames} = \mathbf{1.49 \text{ Gigabytes/sec per camera}}$$
  $$\text{DDR Ingestion Writes} = 1.49\text{ GB/s} \times 4 \text{ cameras} = \mathbf{5.97 \text{ Gigabytes/sec}}$$

---

### 2. NPU Layer Weight Preloading (The Stream Squeeze)
To run a small, dedicated robotics transformer (e.g. **800 Megabytes** total size, 40 layers, each layer = **20 MB**):
* To maintain low battery latency, our target decoding speed is **100 Tokens/second**.
* To generate 100 tokens per second, the NPU **must load the entire 800 MB model 100 times per second autoregressively!**
* **DDR Read Bandwidth Sizing:**
  $$\text{NPU Weight Read BW} = 800\text{ MB} \times 100\text{ Tokens/s} = \mathbf{80.0 \text{ Gigabytes/sec}}$$
* This consumes **`58.6%`** of the total LPDDR5X bandwidth.

---

### 3. Why Activations & KV-Caches Have ZERO DDR Traffic
* For a 2048-token context window (typical for immediate robotic environment memory):
  - Total KV-cache is small ($\approx 24\text{ MB}$).
  - Slicing this KV-cache over our grid PMUs using **INT4 AGU compression** slashes the active footprint to **`6.0 Megabytes`**.
  - Since our NPU has **32 Megabytes** of tightly-coupled, low-leakage on-chip SRAM, **the active activations and KV-caches reside 100% on-chip inside the local PMUs**.
  - No activations are ever spilled or reloaded from DDR, reducing activation DDR traffic to a pristine **`0.00 GB/s`**!

---

## Section 4: Target DDR Capacity Sizing

To host the operating system (RTOS/Ubuntu-RT), ROS2 control loops, LIDAR spatial point maps, and the active LLM control weights, the SoC is configured with:

* **SoC LPDDR5X Capacity:** **`16 Gigabytes`** (dual-die package).

### Memory Space Allocation map:
1. **Robotic AI Weights (Static pin):** **1.0 Gigabyte** (Saves model weights in memory for instant NPU layer preloading).
2. **LiDAR Voxel & SLAM Maps (Dynamic):** **4.0 Gigabytes** (Assigned to GPU for spatial navigation).
3. **ROS2 & RTOS Operating Space:** **4.0 Gigabytes** (Assigned to CPU for general compute).
4. **Camera Sensor Frame Buffers:** **2.0 Gigabytes** (Shared circular ingestion buffers).
5. **System Headroom (Unallocated safety):** **5.0 Gigabytes**.

---
*Report compiled, micro-modeled, and finalized by the Dual-Tier Co-Design Validation Group.*
