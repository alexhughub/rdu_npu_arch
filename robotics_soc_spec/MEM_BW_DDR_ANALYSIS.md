# Memory Bandwidth & Capacity Sizing Analysis: Robotics SoC
**Revision:** v1.1 (Consolidated RTL Design Baseline with All Aux IPs)

This document evaluates the physical memory bandwidth (DDR) and capacity requirements of our humanoid robotics SoC under peak concurrent execution loops, incorporating our newly specified auxiliary IP blocks (Video Decoder, CV Processor, Display Engine, and the Inter-IP Fabric NoC).

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

| IP Block Module | Primary Memory Transaction | Data Format | Active DDR Read/Write | DDR BW Duty Cycle |
| :--- | :--- | :---: | :---: | :---: |
| **Robotics ISP** | 4x 4K Stereo Camera feeds @ 60 FPS | RGB888 / RAW | **`5.97 GB/s`** (Writes) | 4.4% |
| **CV Processor** | Dense optical flow & feature tracking | YUV420 / Vectors | **`3.00 GB/s`** (Reads) | 2.2% |
| **Video Decoder** | Decodes AV1/H.265 diagnostic logs | YUV420 | **`0.90 GB/s`** (Writes) | 0.7% |
| **CPU Complex** | ROS2 planners, IK Solvers, Trajectories| FP32 / Scalars | **`12.00 GB/s`** (R/W) | 8.8% |
| **Robotics GPU** | LiDAR Voxel maps & 3-D Occupancy SLAM | FP32 Vectors | **`20.00 GB/s`** (R/W) | 14.7% |
| **NPU Weight Stream** | Autoregressive Layer-by-Layer Preload | FP8 Weights | **`80.00 GB/s`** (Reads) | 58.6% |
| **NPU Activations** | Tightly coupled S-tiling loops | INT4 (PMU Cached) | **`0.00 GB/s`** (On-Chip) | **0.0% (Zero-Spill)** |
| **Display Engine** | Diagnostic frame output (1080p@60) | RGB888 | **`0.40 GB/s`** (Reads) | 0.3% |
| **System Overhead** | PCIe peripheral buffer & Telemetry | Controls | **`1.00 GB/s`** | 0.7% |
| **Total Co-execution**| **All IP blocks active concurrently** | ? | **`123.27 GB/s`** | **90.3% (Safe margin)**|

* **DDR Scheduling Room:** Slicing all concurrent IP memory requests sums to exactly **`123.27 GB/s`** (**90.3%** of our peak LPDDR5X-8533 bandwidth). This leaves a comfortable **`13.25 GB/s` (9.7%)** unallocated safety margin to absorb DRAM bank precharge delays, page conflict penalties, and interface commands.

---

## Section 3: Deep Microarchitectural BW Sizing Calculations

```
                     CONCURRENT DDR BANDWIDTH ALLOCATION
                     
   LPDDR5X-8533 Interface Bus (136.5 GB/s Peak)
   |
   +===[  58.6% ]===> NPU Layer Weight Streaming (80.0 GB/s)
   |
   +===[  14.7% ]===> GPU Spatial Occupancy mapping (20.0 GB/s)
   |
   +===[   8.8% ]===> CPU Navigation & ROS2 Planners (12.0 GB/s)
   |
   +===[   4.4% ]===> ISP Camera Ingestion Writes (5.97 GB/s)
   |
   +===[   2.2% ]===> CV Processor Optical Flow Reads (3.0 GB/s)
   |
   +===[   0.7% ]===> Video Decoder YUV Writes (0.90 GB/s)
   |
   +===[   0.3% ]===> Display Diagnostic Reads (0.40 GB/s)
   |
   +===[   0.7% ]===> Telemetry & PCIe Buffers (1.0 GB/s)
   |
   +===[   9.7% ]===> Unallocated DDR Safety Headroom (13.25 GB/s)
```

### 1. Robotics ISP Camera Ingestion (DDR Write Sizing)
* **The Ingest:** 4x 4K Stereo cameras running at 60 FPS uncompressed.
* **RAW Sensor Input:** $3840 \times 2160 \times 10\text{ bits} \times 60\text{ FPS} = 4.97\text{ Gbps} = 622\text{ MB/s}$ per camera. Slices 4 cameras = **`2.49 GB/s`** into the ISP.
* **YUV/RGB Convert Write:** The ISP converts RAW pixels to uncompressed RGB888 for the vision transformers:
  $$\text{RGB888 4K @ 60 FPS} = 3840 \times 2160 \times 3\text{ Bytes} \times 60 \text{ frames} = \mathbf{1.49 \text{ Gigabytes/sec per camera}}$$
  $$\text{DDR Ingestion Writes} = 1.49\text{ GB/s} \times 4 \text{ cameras} = \mathbf{5.97 \text{ Gigabytes/sec}}$$

---

### 2. CV Processor (DDR Read Sizing)
* **The Ingest:** Reads uncompressed camera frames from DDR to execute real-time optical flow and feature tracking:
  $$\text{YUV420 4K @ 60 FPS} = 3840 \times 2160 \times 1.5\text{ Bytes} \times 60 \text{ frames} = \mathbf{746.49 \text{ MB/s per camera}}$$
  $$\text{DDR Feature Reads} = 746.49\text{ MB/s} \times 4 \text{ cameras} = \mathbf{2.98 \text{ GB/s} \approx 3.0 \text{ Gigabytes/sec}}$$

---

### 3. Video Decoder (DDR Write Sizing)
* **The Ingest:** Reads AV1/H.265 compressed diagnostic data and writes uncompressed YUV420 frames back to DDR:
  $$\text{Decoded Frame Write} = 1 \text{ stream} \times 746.49\text{ MB/s} \times 1.2 \text{ (Ref Frames)} \approx \mathbf{0.90 \text{ Gigabytes/sec}}$$

---

### 4. Display Engine (DDR Read Sizing)
* **The Ingest:** Reads uncompressed diagnostic frame buffer (e.g., standard 1080p@60 FPS) to feed HDMI 2.1 / eDP:
  $$\text{Display Read} = 1920 \times 1080 \times 3\text{ Bytes} \times 60\text{ frames} = 373,248,000\text{ Bytes/s} \approx \mathbf{0.40 \text{ Gigabytes/sec}}$$

---

### 5. NPU Layer Weight Preloading (The Stream Squeeze)
To run a small, dedicated robotics transformer (e.g. **800 Megabytes** total size, 40 layers, each layer = **20 MB**):
* To maintain low battery latency, our target decoding speed is **100 Tokens/second**.
* To generate 100 tokens per second, the NPU **must load the entire 800 MB model 100 times per second autoregressively!**
* **DDR Read Bandwidth Sizing:**
  $$\text{NPU Weight Read BW} = 800\text{ MB} \times 100\text{ Tokens/s} = \mathbf{80.0 \text{ Gigabytes/sec}}$$
* This consumes **`58.6%`** of the total LPDDR5X bandwidth.

---

### 6. Inter-IP Fabric NoC Bandwidth
* **The Squeeze:** The Inter-IP Fabric NoC is the on-chip coherent ring bus. It must carry the aggregate memory requests from all on-chip IPs to the LPDDR5X memory controller.
* **Sizing Specifications:** Sized as a **384-bit wide parallel bus** operating sync-clocked at **1.2 GHz**. 
* **Fabric NoC Link Bandwidth:**
  $$\text{Fabric NoC BW} = 384\text{ bits} \times 1.2\text{ GHz} = 460.8\text{ Gbps} = \mathbf{57.6 \text{ Gigabytes/sec per directional direction}}$$
* Multiple coherent rings provide an aggregate fabric bandwidth of **`230.4 GB/s`**, comfortably providing over 1.7x headroom for the maximum concurrent memory interface traffic of **`123.27 GB/s`**!

---
*Report compiled, micro-modeled, and finalized by the Dual-Tier Co-Design Validation Group.*
