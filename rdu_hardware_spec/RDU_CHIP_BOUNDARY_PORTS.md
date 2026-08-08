# Chip Boundary Port & Signal Definition: Next-Gen RDU

This document defines the physical boundary ports, signal groups, and pin-out parameters of the next-generation **1.24 PFlops Reconfigurable Dataflow Unit (RDU)**. 

---

## 1. Physical Chip Interface Architecture

The next-gen RDU package uses a High-Density Ball Grid Array (BGA) footprint with **3,120 active pins** (including power and ground rails). The active interfaces are organized into four distinct boundary groups:

```
                       RDU PHYSICAL BOUNDARY PORT PIN-OUT
                       
                 +---------------------------------------------+
                 |             Power & Ground Rails            |
                 |                 1,400 pins                  |
                 +---------------------------------------------+
   +-------------+                                             +-------------+
   |  6x HBM3    |                                             |   8x ICL    |
   |  PHY Ports  |                   RDU DIE                   | High-Speed  |
   | 1,024 pins  |                  (495 mm²)                  |  boundary   |
   +-------------+                                             |  512 pins   |
                 +---------------------------------------------+
                 |            Control & Clock Ports            |
                 |                  184 pins                   |
                 +---------------------------------------------+
```

---

## 2. Boundary Port Signal Group Definitions

### Group A: Inter-Chip Links (ICL) - 512 Pins
Responsible for direct, multi-socket scaling (Tensor Parallelism TP=1 to TP=8) across physical RDUs. 
* **Port Count:** **8 independent ports** (ICL[0:7]).
* **Signal Type:** Differential SerDes lanes running at **`32 Gbps`** per lane.
* **Signals per Port:** 32 differential pairs (16 TX pairs, 16 RX pairs) = 64 pins per port.
* **ICL Aggregate BW:** **150 Gigabytes/sec** bidirectional per port (1.2 TB/s aggregate per chip).

| Port Name | Direction | Pin Count | Protocol / Standard | Micro-Arch Function |
| :--- | :---: | :---: | :--- | :--- |
| **ICL_TX_P[0:7][0:15]** | Output | 128 | Custom Differential SerDes | Direct Torus NoC packet forwarding (Positive leg) |
| **ICL_TX_N[0:7][0:15]** | Output | 128 | Custom Differential SerDes | Direct Torus NoC packet forwarding (Negative leg) |
| **ICL_RX_P[0:7][0:15]** | Input | 128 | Custom Differential SerDes | Direct Torus NoC packet forwarding (Positive leg) |
| **ICL_RX_N[0:7][0:15]** | Input | 128 | Custom Differential SerDes | Direct Torus NoC packet forwarding (Negative leg) |

---

### Group B: HBM3 PHY Interface - 1,024 Pins
Connects the on-chip silicon interposer directly to the 6 physical HBM3 stacks.
* **Port Count:** **6 HBM3 Channels** (HBM3[0:5]).
* **Signal Type:** Single-ended and differential high-speed memory interface.
* **Data Rate:** **`6.4 Gbps`** per signal line.
* **Aggregate Memory BW:** **`2.4 TB/s`** off-chip memory bandwidth.

| Port Name | Direction | Pin Count | Standard | Micro-Arch Function |
| :--- | :---: | :---: | :--- | :--- |
| **HBM_DQ[0:5][0:127]** | Bidirect | 768 | HBM3 JEDEC Spec | 128-bit wide data bus per stack channel |
| **HBM_WDQS[0:5][0:15]**| Input/Out| 96 | HBM3 JEDEC Spec | Differential write strobe lines |
| **HBM_RDQS[0:5][0:15]**| Output | 96 | HBM3 JEDEC Spec | Differential read strobe lines |
| **HBM_A_C[0:5][0:10]** | Output | 64 | HBM3 JEDEC Spec | Command/Address bus lines |

---

### Group C: Control, System & Clock Interfaces - 184 Pins
Responsible for boot configurations, diagnostic telemetry, and system-level synchronization.
* **Clocking Paradigm:** System operates on a **Single-Source Reference Clock (100 MHz)**. The internal 1.35 GHz clock is generated via on-chip **Phase-Locked Loops (PLLs)**.

| Port Name | Direction | Pin Count | Signal Standard | Micro-Arch Function |
| :--- | :---: | :---: | :--- | :--- |
| **REF_CLK_P / N** | Input | 2 | Differential LVDS | 100 MHz reference clock input |
| **SYS_RST_N** | Input | 1 | CMOS 1.8V | Global asynchronous system hardware reset |
| **JTAG_TCK/TDI/TDO/TMS**| Joint | 4 | IEEE 1149.1 JTAG | Bound scan debugging and diagnostic tap |
| **PCIe_REF_CLK** | Input | 2 | Differential LVDS | Host PCIe reference clock |
| **PCIe_TX / RX[0:15]** | Joint | 64 | PCIe Gen5 SerDes | 16-lane PCIe Host link interface (64 GB/s) |
| **GPIO[0:31]** | Bidirect | 32 | CMOS 1.8V | Reconfigurable auxiliary telemetry control pins |
| **I2C_SCL / SDA** | Bidirect | 2 | CMOS 1.8V | Board-level thermal and voltage sensor loop |
| **PMBUS_CLK / DATA** | Bidirect | 2 | CMOS 1.8V | Power supply telemetry control |
| **AUX_PINS** | Joint | 75 | Low-speed I/O | Static boot straps and pull-ups |

---

### Group D: Power & Ground Rails - 1,400 Pins
Due to the custom 1.35 GHz frequency scaling, the chip power rails are partitioned to isolate sensitive memory and clock domains from compute noise.

| Rail Name | Target Voltage | Pin Count | Target Function | Target Current |
| :--- | :---: | :---: | :--- | :---: |
| **VDD_CORE** | **`0.75V`** | 600 | Power supply for PCU logic and tile grid | 240 Amps |
| **VDD_SRAM** | **`0.85V`** | 200 | Dedicated supply for 128MB 8T PMU SRAM blocks | 60 Amps |
| **VDD_HBM** | **`1.10V`** | 200 | Dedicated supply for 6x HBM3 physical stacks | 110 Amps |
| **VDD_PLL** | **`1.80V`** | 50 | Isolated analog supply for on-chip clock PLLs | 2 Amps |
| **VSS_GND** | **`0.00V`** | 350 | Unified reference system ground | 450 Amps |

---
*Report compiled and drafted as the Downstream Micro-Arch baseline.*
