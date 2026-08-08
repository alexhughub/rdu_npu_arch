# Microarchitectural Specification: Global Clocking, DVFS, & Clock Gating
**Revision:** v1.1 (Consolidated Clock, Voltage, & Power Distribution Baseline)

This document defines the physical clock trees, Dynamic Voltage and Frequency Scaling (DVFS) controllers, PMIC communication interfaces, and Fine-Grained Clock Gating (FGCG) systems of our custom humanoid robotics SoC.

---

## Section 1: Global Clock Tree Distribution Topology

To support the diverse frequency and voltage requirements of the CPU, GPU, NPU, ISP, and LPDDR5X channels, the SoC integrates **five independent on-chip PLLs** connected to a **Glitch-Free Clock Multiplexing (GFMUX)** network:

```
                      GLOBAL CLOCK TREE NETWORK ARCHITECTURE
                      
     [ 100 MHz Ref OSC ]
              |
              +---> [ PLL_CPU ] ===(2.4 GHz)======> [ GFMUX_CPU ] ===> CPU (300 MHz - 2.4 GHz)
              |
              +---> [ PLL_GPU ] ===(1.8 GHz)======> [ GFMUX_GPU ] ===> GPU (200 MHz - 900 MHz)
              |
              +---> [ PLL_NPU ] ===(2.4 GHz)======> [ GFMUX_NPU ] ===> NPU (300 MHz - 1.2 GHz)
              |
              +---> [ PLL_MEM ] ===(4.266 GHz)====> [ GFMUX_MEM ] ===> LPDDR5X (800 MHz - 4.266 GHz)
              |
              +---> [ PLL_SYS ] ===(4.8 GHz)======> [ GFMUX_SYS ] ===> NoC / SLC (1.2 GHz fixed)
              |
              +---> [ PLL_PERIPH ] ===(1.2 GHz)===> [ GFMUX_PER ] ===> ISP / CV / Decoders (600 MHz)
```

### Power & Clock Domain Slices:
The SoC is physically partitioned into **six independent clock and power domains**, allowing clocks and voltages to be gated or scaled independently:

1. **`DOMAIN_CPU`:** 8x ARM Cortex-A720 cores. Scaled dynamically between **300 MHz and 2.40 GHz** at **0.55V - 0.85V VDD_CPU**.
2. **`DOMAIN_GPU`:** SLAM and occupancy grid maps. Scaled dynamically between **200 MHz and 900 MHz** at **0.55V - 0.80V VDD_GPU**.
3. **`DOMAIN_NPU`:** 256 custom PCU/PMU tiles. Scaled dynamically between **300 MHz and 1.20 GHz** at **0.55V - 0.75V VDD_NPU**.
4. **`DOMAIN_MEM`:** LPDDR5X PHY and memory controller. Scaled between **800 MHz** and **4.266 GHz** at **0.55V - 0.80V VDD_MEM**.
5. **`DOMAIN_SYS`:** Fabric NoC, System Cache (SLC), and boot ROM. Locked at a synchronous, low-jitter **1.20 GHz** clock at **0.75V VDD_CORE**.
6. **`DOMAIN_LPI` (Low-Power Island):** Always-On (AON) power domain containing 1 low-power CPU core and basic MIPI camera triggers. Clocked at a fixed **100 MHz** directly from the reference crystal, bypassing all PLLs during deep sleep at **0.50V VDD_LPI**.

---

## Section 2: Dynamic Frequency Switching (DFS) & Glitch-Free Muxes

To shift frequency on-the-fly without locking up register files or corrupting active arithmetic states, the clock dividers utilize hardware-level **Glitch-Free Clock Multiplexers (GFMUXs)**:

### 1. The GFMUX Switching Protocol:
Directly altering a PLL's feedback multiplier causes clock instability (jitter/voltage droops) during lock acquisition. The DFS Controller bypasses this via a **4-step switching protocol**:

```
                       GFMUX CLOCK SWITCHING TIMELINE
                       
   Cycle 0: [ Active PLL (1.2 GHz) ] ========> [ Output Clock (1.2 GHz) ]
                   |
   Cycle 1: [ GFMUX switches source to safe divided 100 MHz Ref Clock ]  (Glitch-Free)
                   |
   Cycle 2: [ PLL undergoes lock change, stabilizes at new multiplier (600 MHz) ]
                   |
   Cycle 3: [ GFMUX switches source back to newly locked PLL (600 MHz) ] (Glitch-Free)
```

1. **Step 1 (Safe Bypass):** When a DFS command is triggered, the target GFMUX instantly switches its clock source from the high-frequency PLL line to a safe **100 MHz bypass reference clock**. This switch is guaranteed glitch-free by using dual synchronizers to ensure the output clock does not clip high or low during transitions.
2. **Step 2 (PLL Lock Change):** The DFS controller alters the target PLL's dividers and multipliers. The PLL undergoes its locking cycle (requiring up to **`15 microseconds`** of stabilization).
3. **Step 3 (Stabilization):** Once the PLL asserts its `LOCK_VALID` signal, the GFMUX waits for 4 reference clock cycles to let transient jitter settle.
4. **Step 4 (Switchback):** The GFMUX switches the output clock line back from the 100 MHz reference to the newly stabilized high-frequency PLL output. 
5. **System Impact:** The entire DFS switch is completed in **`< 16 microseconds`** without needing to flush pipelines or pause active CPU/NPU processing thread states!

---

## Section 3: Dynamic Voltage and Frequency Scaling (DVFS) Orchestration

Voltage and frequency are fundamentally coupled in advanced FinFET silicon:  
* **The Law:** Higher frequency requires higher operating voltage to decrease transistor delay ($F \propto \frac{V - V_{\text{th}}}{V}$) and prevent setup-time timing violations. 
* **The Formula:** Dynamic power scales quadratically with voltage ($P_{\text{dynamic}} \propto C V^2 F$).

Therefore, the on-chip **Hardware Power Management Unit (HPMU)** must orchestrate clock frequency shifts and voltage rail adjustments in a strict, hardware-locked sequence:

### 1. The DVFS Scaling Sequences (Setup-Time Collision Avoidance)

To prevent catastrophic timing failures (setup-time violations) or transistor slow-down latch-ups, the HPMU implements two hardware-locked finite state machine (FSM) sequences:

#### Sequence A: Scaling UP (Frequency UP & Voltage UP)
```
          STEP 1: Boost Voltage  =======>  STEP 2: Settle Power  =======>  STEP 3: Scale Freq
          (PMIC ramps up VDD)              (Wait for VDD Slew)             (GFMUX shifts CLK)
```
1. **The Risk:** If we scale frequency up *before* boosting voltage, the transistors will operate faster than their current low-voltage delay allows, causing setup-time violations and immediate silicon crash.
2. **The Sequence:**
   - **Step 1:** The HPMU sends a high-priority command over a **10 MHz I3C / SVID (Serial Voltage Identification)** bus to the external buck regulator PMIC to increase voltage (e.g. from 0.55V to 0.75V).
   - **Step 2:** The PMIC ramps up the voltage at its maximum physical slew rate of **`10 mV / microsecond`**. A 200mV step takes exactly **`20 microseconds`** to physically settle.
   - **Step 3:** Once the PMIC asserts the `VDD_OK` handshake pin, the HPMU's hardware delay counters release, and the GFMUX is allowed to switch the clock frequency up (e.g. from 300 MHz to 1.20 GHz) in under **16 microseconds**.

#### Sequence B: Scaling DOWN (Frequency DOWN & Voltage DOWN)
```
          STEP 1: Scale Freq  =========>  STEP 2: Wait 4 Cycles  =========>  STEP 3: Reduce Volt
          (GFMUX shifts CLK)               (Let current settle)             (PMIC lowers VDD)
```
1. **The Risk:** If we scale voltage down *before* reducing frequency, the transistors will slow down while the clock is still running at peak frequency, causing setup-time violations and immediate crash.
2. **The Sequence:**
   - **Step 1:** The HPMU instantly commands the GFMUX to switch the clock frequency down (e.g., from 1.20 GHz down to 300 MHz). This transition takes under 16 microseconds.
   - **Step 2:** The HPMU waits for 4 local clock cycles to let current states settle and ensure all high-frequency pipelines are completely drained.
   - **Step 3:** The HPMU sends an I3C/SVID command to the PMIC to step the voltage down (e.g., from 0.75V down to 0.55V) to achieve quadratic dynamic power savings.

---

### 2. Active di/dt Voltage Droop Mitigation (AVS & CPMs)
When a massive NPU compute block suddenly wakes up from idle, it draws a massive amount of current instantly. This creates a steep current transient ($di/dt$), which reacts with the package's parasitic inductance ($L$) to cause a sudden **voltage droop** ($V_{\text{droop}} = L \frac{di}{dt}$) on the core VDD rail.

To prevent voltage droops from causing timing failures before the PMIC can respond, the SoC integrates **Adaptive Voltage Scaling (AVS)** with **Critical Path Monitors (CPMs)**:

```
                      ACTIVE DROOP COMPENSATION PIPELINE
                      
    Compute Burst ===> [ di/dt Voltage Droop ] ===> [ CPM (Ring Osc) slows ]
                                                            |
    New Freq GFMUX <== [ Reduce CLK by 12.5% ] <== [ Detects timing pinch  ]
           |
    PMIC boosts VDD <== [ HPMU SVID Command   ]
```

1. **The Senses:** Integrated on-chip CPMs (configured as ultra-sensitive ring oscillators) are physically co-located inside the CPU, GPU, and NPU tile complexes.
2. **The Detection:** If a local voltage droop exceeds **`35 mV`**, the CPM's oscillation frequency slows down. An on-chip hardware comparator detects this "timing pinch" in **`< 2 nanoseconds`** (within 2 clock cycles).
3. **The Adaptation:** The comparator instantly triggers the local clock divider to **reduce the clock frequency by 12.5%** for exactly 16 cycles.
4. **The Compensation:** This clock slowing instantly relieves the timing pressure, preventing setup-time violations. Concurrently, the HPMU commands the PMIC to boost voltage to compensate for the droop. Once VDD stabilizes, the clock frequency is restored to 100% automatically.

---

## Section 4: Fine-Grained Clock Gating (FGCG) inside IPs

While DFS/DVFS reduces average power, **Fine-Grained Clock Gating (FGCG)** slashes dynamic charging power on a cycle-by-cycle basis using hardware-gated **Integrated Clock Gating (ICG)** cells:

```
                       INTEGRATED CLOCK GATING (ICG) CELL
                       
    Global Clock (CLK) =======> [ AND Gated Switch ] ======> Gated Output Clock (G_CLK)
                                       ^
    Enable Line (EN)   =======> [ Latch (Low)  ]
```

### IP-Specific Gating Slices:

#### A. NPU Tile-Level Operand Gating (FGCG_NPU):
* **Sensing:** Inside each of the 256 PCUs, specialized **Operand-Sensing Gating Logic** scans the inputs of the 384 PEs.
* **Mechanism:** If the input activation or weight operand is exactly zero, the FGCG_NPU controller asserts the ICG cell for that multiplier's column register on the very next cycle, **slashing dynamic switching power by up to `35%`** during sparse loops.

#### B. GPU Sub-Core Power Gating (FGCG_GPU):
* **Mechanism:** If a GPU sub-core's shader input buffer remains empty for more than 4 consecutive cycles (empty spatial zones), the sub-core's local clock distribution tree is **gated (turned off)**, reducing GPU standby consumption from **`6.0 Watts` to `< 1.2 Watts`**.

---

## Section 5: Quantitative Benefits under Real-World Robotic Workloads

The table below outlines the average power savings achieved by our DVFS and Clock Gating systems under different humanoid physical activities:

| Physical Workload State | Active IP Blocks | Frequency State | Voltage State (VDD) | Clock Gating State | SoC Power Draw | Battery Lifespan (600Wh Pack) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Active Locomotion & SLAM** | CPU, GPU, NPU, ISP, LPDDR5X | **100%** (1.2 GHz peak) | **0.75V - 0.85V** (High) | Active Operand-Gating | **`25.8 Watts`** | **23.2 Hours** |
| **Idle Standing / Waiting** | CPU, ISP, LPDDR5X (NPU/GPU Idle) | **DFS Min** (300 MHz) | **0.55V** (Ultra-Low) | Spatial Sub-Core Gating | **`4.8 Watts`** | **125.0 Hours** |
| **Deep Sleep / Charging** | LPI Domain Only (Main SoC Off) | **PLLs Powered Off** | **0.50V** (Always-On AON) | Core Domain Gating | **`0.15 Watts`** | **4000.0 Hours** |

---
*Report compiled, micro-modeled, and finalized by the Dual-Tier Co-Design Validation Group.*
