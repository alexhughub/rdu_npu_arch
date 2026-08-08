# Microarchitectural Justification: NPU SRAM vs. System-Level Cache (SLC) Sizing

This document provides a rigorous, first-principles mathematical and physical justification for why the dedicated **NPU SRAM (32 MB)** is physically larger than the shared **System-Level Cache (SLC, 16 MB)** inside our edge humanoid robotics SoC.

---

## 1. The Core Sizing Paradox

At first glance, a system engineer might ask:  
* **The Question:** Why is the dedicated NPU SRAM (**`32 MB`**) twice the size of the shared System-Level Cache (**`16 MB`**)? Shouldn't a shared system cache be larger than a single processor's local cache?
* **The Architectural Law:** The size of any on-chip cache is dictated purely by the **Active Working Set Sizing** of its target workloads. Because the NPU processes massive deep learning weight layers (Megabytes), while the CPU/GPU process sparse spatial/scalar structures (Kilobytes), the NPU requires a vastly larger on-chip memory footprint to prevent performance-killing DRAM spills.

---

## 2. NPU SRAM Sizing Calculations (Why 32 MB is the Minimum)

To execute our target 800 MB robotics transformer model (40 layers, each layer = **20 MB**) at a blinding, real-time speed of **100 Tokens/second** under $TP=1$:

$$\text{NPU SRAM Requirements} = \text{Double-Buffered Weights} + \text{Active Activations} + \text{Compressed KV-Cache}$$

### A. Double-Buffered Weight Prefetch (20.0 Megabytes):
* To achieve zero-latency stalls, while the PCU ALUs are computing on Layer $k$, the NPU's prefetch engine must asynchronously load the weights of Layer $k+1$ from DDR.
* By sequence-tiling the layers, the NPU streams weights in half-layer partitions.
  - Active Weight Buffer = **10.0 MB** (currently being read by PCUs).
  - Prefetch Weight Buffer = **10.0 MB** (currently being written by DDR prefetcher).
  - Total Weight Footprint = **`20.0 MB`**.

### B. Active Input Activation/Query Tensor (4.0 Megabytes):
* The active intermediate activations for a single sequence chunk ($S_{\text{micro}} = 256$ tokens) at standard FP16 requires:
  $$\text{Activation Footprint} = 256 \text{ tokens} \times 8192 \text{ hidden dim} \times 2 \text{ bytes (FP16)} = \mathbf{4,194,304 \text{ Bytes} \approx 4.0\text{ MB}}$$

### C. Compressed Local KV-Cache (6.0 Megabytes):
* For a 2048-token local robotics environment memory context, the raw KV-cache is 24 MB.
* Under our **INT4 AGU hardware compression**, this footprint is slashed by 4x down to:
  $$\text{Compressed KV-Cache Footprint} = \frac{24\text{ MB}}{4} = \mathbf{6.0\text{ MB}}$$

### D. Auxiliary Control & Local Registers (2.0 Megabytes):
* Dedicated to storing instruction program blocks, local PCU register files, and boundary NoC buffers.

$$\text{Total NPU SRAM Required} = 20.0\text{MB (Weights)} + 4.0\text{MB (Activations)} + 6.0\text{MB (KV-Cache)} + 2.0\text{MB (Aux)} = \mathbf{32.0\text{ Megabytes}}$$

* **The Squeeze:** If we sized the NPU SRAM any smaller (e.g. 16MB), the NPU would be **forced to spill weights or activations back to DDR**, triggering a catastrophic Memory Wall bottleneck and dropping token generation speeds.

---

## 3. System-Level Cache Sizing Calculations (Why 16 MB is the Sweet Spot)

The shared SLC buffers high-frequency data from the CPU (ROS2 planners), GPU (SLAM occupancy grids), and ISP (camera pixel pipes) to prevent redundant DDR read/write operations.

Unlike the NPU, the active working sets of these standard IPs are extremely small:

### A. ISP Camera Frame Buffering (6.2 Megabytes):
* The ISP receives pixel streams from 4x 4K stereo cameras. However, the ISP does *not* need to cache the entire uncompressed 4K frame (1.49 GB/s) on-chip.
* It operates as a streaming line-buffer, caching only a few active horizontal scanlines during de-mosaicing before writing them out. 
* To buffer one active YUV420 frame slice per camera concurrently:
  $$\text{ISP Cache Footprint} = 4 \text{ cameras} \times 1.5\text{ MB per frame slice} = \mathbf{6.0\text{ MB}}$$

### B. GPU LiDAR SLAM Voxel Maps (0.5 Megabytes):
* The GPU processes a real-time local occupancy grid of $100 \times 100 \times 50$ voxels at 8-bit density:
  $$\text{Voxel Cache Footprint} = 100 \times 100 \times 50 \times 1\text{ Byte} = 500,000\text{ Bytes} \approx \mathbf{0.5\text{ MB}}$$

### C. CPU ROS2 Planners & Kinematics (1.5 Megabytes):
* The active ROS2 trajectory matrices and planning variables are sparse, requiring less than **`1.5 MB`** of active cache space.

### D. System Headroom & Shared Buffer (8.0 Megabytes):
* Shared dynamically as a scratchpad for inter-IP communication and PCIe peripheral transfers.

$$\text{Total SLC Required} = 6.0\text{MB (ISP)} + 0.5\text{MB (GPU)} + 1.5\text{MB (CPU)} + 8.0\text{MB (Headroom)} = \mathbf{16.0\text{ Megabytes}}$$

* **The Area Squeeze:** Sizing the SLC larger than 16MB (e.g. to 32MB) would represent a **waste of precious silicon area**. The CPU, GPU, and ISP do *not* have massive, monolithic weight-and-activation streaming dependencies like the NPU. 
* Sizing the SLC at **16 MB** keeps the TSMC 4nm die size compact and highly cost-optimal.

---

## 4. Master Co-Design Summary

The RDU / SoC memory hierarchy is sized with mathematical precision to achieve optimal throughput-per-watt:
1. **NPU SRAM is sized at 32 MB** because deep learning layers have massive active working sets (20 MB weights + 10 MB activations).
2. **Shared SLC is sized at 16 MB** because standard robotics planning, mapping, and vision streams have sparse active working sets.

---
*Report compiled, micro-modeled, and finalized by the Dual-Tier Co-Design Validation Group.*
