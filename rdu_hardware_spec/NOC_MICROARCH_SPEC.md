# Microarchitectural Specification: Network-on-Chip (NoC) Router
**Revision:** v1.0 (Downstream RTL Design Baseline)

This document defines the physical interface, virtual channels, routing protocols, registers, and power-saving mechanisms of the **Network-on-Chip (NoC)** router inside each homogeneous tile of the RDU.

---

## 1. Top-Level Block Diagram & Interface Slices

Each of the 1,024 tiles contains a dedicated **5-Port NoC Router**. Sockets and tiles are interconnected in a **2D Torus** topology over five physical routing ports:

```
                      NoC ROUTER MICROARCHITECTURAL DIAGRAM
                      
                               +------------------+
                               |     North Port   | (256b + credit handshaking)
                               +--------+---------+
                                        |
       +------------------+    +--------+---------+    +------------------+
       |     West Port    |<==>|  5x5 Non-blocking|<==>|     East Port    |
       | (256b + credit)  |    | Crossbar Switch  |    | (256b + credit)  |
       +------------------+    +--------+---------+    +------------------+
                                        |
                        +---------------+---------------+
                        | Local Port (PMU)              |
                        | (256b parallel link)          |
                        v                               v
                  +-----+-------------------------------+-----+
                  |                  NoC ROUTER               |
                  |                                           |
                  |     +-------------------------------+     |
                  |     |  Arbiter & Routing Controller |     |
                  |     +-------------------------------+     |
                  |                                           |
                  +---------------------+---------------------+
                                        |
                               +--------v---------+
                               |     South Port   | (256b + credit handshaking)
                               +------------------+
```

### Physical Sizing Parameters (7nm Node):
* **Physical Area:** **`~0.012 mm²`** per NoC Router.
* **Active Power TDP:** **`~18 mW`** per Router at 1.35 GHz.
* **Link Bandwidth (Bidirectional):** **`43.2 Gigabytes/sec per link direction`**.

---

## 2. Router Ports & Virtual Channel Slicing

To prevent hardware deadlocks and separate different types of traffic on the mesh:

### A. Sub-Module Division:
1. **5-Port Configuration:** North, South, East, West, and Local (connecting the local tile's PMU).
2. **Double-Wide Links:** Physical link width is **256-bit wide** parallel lines sync-clocked at **1.35 GHz**.
3. **Flit FIFO Queues:** Sized as **12-flit deep FIFO buffers** per port to absorb congestion.
4. **Virtual Channels (VC):** Each physical port is sliced into **4 independent Virtual Channels (VC[0:3])**:
   - **`VC0` (Compute Dataflow):** Dedicated to high-speed activation/query packet transfers.
   - **`VC1` (Weight Streaming):** Dedicated to background HBM weight-prefetch streaming.
   - **`VC2` (KV-Cache Transfer):** Dedicated to inter-socket/inter-tile KV history transfers.
   - **`VC3` (Control & Synchronization):** Dedicated to hardware barriers, credit syncs, and system-level messages.

---

## 3. Routing Protocol: Torus XY Routing & Spatial Multicasting

### A. XY Dimension-Ordered Routing (DOR):
* Implements a deterministic **XY Routing Protocol** to guarantee deadlock-free packet routing.
* Packets are always routed along the X-dimension (East/West) first.
* Once the target X-coordinate is reached, the packet is routed along the Y-dimension (North/South) to the local tile.

### B. Spatial Multicasting (Broadcasting):
* Sockets can execute **hardware-accelerated multicasting**.
* Allows a single weight or activation frame to be copied and broadcasted to multiple rows of tiles simultaneously over a single link cycle!
* **The Slicing Benefit:** During the weight prefetch loop, HBM streams weights onto the NoC once, and the NoC routers dynamically clone and distribute the weight flits to all 32 tiles in a row simultaneously, dropping HBM bisection congestion to near-zero!

---

## 4. NoC Bus Protocol: Advanced Spatial Bus Protocol (ASBP)

The RDU NoC operates on a custom, credit-based **Advanced Spatial Bus Protocol (ASBP)**:

* **Flit-based packetization:** Slices data packets into **256-bit Flits** (1 Head flit, 10 Body flits, 1 Tail flit).
* **Credit-Based Flow Control:** Routers only transmit a flit if the downstream router has returned a buffer-free credit. This prevents FIFO overflows and hardwired packet loss.
* **Direct ICL boundary bridging:** The boundary transceivers translate ASBP flits directly into inter-socket differential SerDes frames with zero protocol translation overhead, maintaining direct NoC extension across all 8 sockets.

---

## 5. Register Files & Control Registers

Each Router contains configuration and debug registers:

* **`NOC_ROUTE_REG` (32b):** Configures the static XY routing coordinates for the local tile.
  - Bit[5:0]: X-Coordinate.
  - Bit[11:6]: Y-Coordinate.
* **`NOC_CREDIT_COUNT` (16b):** Senses and counts available downstream buffer credits per Virtual Channel.
* **`NOC_MULTI_MASK` (32b):** Configures multicast mask ranges. Indicates which rows/columns of the grid should receive cloned broadcast flits.

---

## 6. Popular Power-Saving Techniques Employed

To prevent high dynamic switching power over the massive grid network, the NoC implements:

1. **Zero-Byte Dataflow Encoding (ZBE):**
   * Senses incoming data flits. If a flit contains null bytes (all zeroes, which occurs frequently in sparse MoE gating calculations or masked attention pads), the router **gates (turns off) the physical wire toggle drivers** for those zeroed byte columns and transmits a 1-byte "zero-mask" flag.
   * This reduces dynamic wire-charging power over long NoC link lines by up to **`40%`**!
2. **Activity-Based Link Power Down:**
   * If a physical NoC link direction is inactive for more than 16 clock cycles (e.g. during a vector compute stall), the link transceivers enter a low-power **standby state**, saving **`60%`** of static link transceiver power.

---
*Report compiled and structured as the Downstream Micro-Arch baseline.*
