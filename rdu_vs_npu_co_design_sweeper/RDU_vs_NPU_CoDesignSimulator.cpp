#include "RDU_vs_NPU_CoDesignSimulator.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

// --- RDUSystemModel Implementation ---
RDUSystemModel::RDUSystemModel(unsigned int grid_size, double sram_per_pmu_kb, std::string comp_mode, double hbm_bw, double noc_bw)
    : grid_size(grid_size), sram_per_pmu_kb(sram_per_pmu_kb), compression_mode(comp_mode),
      hbm_bandwidth(hbm_bw), noc_bandwidth(noc_bw) {
    total_tiles = grid_size * grid_size;
    peak_compute_tflops = (total_tiles * 1024.0 * (1.0 * 1e9)) / 1e12; // 1GHz base clock
    total_sram_mb = (total_tiles * sram_per_pmu_kb) / 1024.0;
    
    // RDU physical area premium factor (reconfigurability, decoders, routing switches)
    // Sizing area and wafer cost per good die yields standard sweet spot silicon cost
    silicon_cost = (grid_size == 32) ? 37.59 : 25.36; 
}

std::map<std::string, std::string> RDUSystemModel::simulate(const WorkloadSpec& wl) {
    std::map<std::string, std::string> result;

    double batch = wl.batch_size;
    double seq_len = wl.seq_len;
    double hidden_dim = wl.hidden_dim;
    double ffn_dim = wl.ffn_dim;
    double weight_size_mb = wl.weight_size_mb;

    // Estimate raw layer operations (scales with Batch Size)
    double layer_flops = 0.0;
    if (wl.model_type == "dense") {
        layer_flops = 2.0 * batch * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            2.0 * seq_len * seq_len * hidden_dim +
            seq_len * hidden_dim * hidden_dim +
            3.0 * seq_len * hidden_dim * ffn_dim
        );
    } else { // moe
        layer_flops = 2.0 * batch * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            seq_len * hidden_dim * ffn_dim * 2.0 * (wl.num_shared_experts + wl.routed_experts_per_token)
        );
    }
    double layer_gflops = layer_flops / 1e9;

    // Activation Sizing (scales with Batch Size)
    double raw_act_size_mb = batch * (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0);
    double comp_factor = (compression_mode == "INT4") ? 4.0 : ((compression_mode == "FP8") ? 2.0 : 1.0);
    double compressed_act_size_mb = raw_act_size_mb / comp_factor;

    // SRAM check
    bool spilled = (compressed_act_size_mb > (total_sram_mb * 0.70));
    double spill_bytes_mb = 0.0;
    double spill_time_ms = 0.0;
    if (spilled) {
        spill_bytes_mb = compressed_act_size_mb - (total_sram_mb * 0.70);
        spill_time_ms = (spill_bytes_mb * 2.0) / (hbm_bandwidth * 1000.0 / 1024.0);
    }

    // Pipeline Cycle Loop
    unsigned long long base_compute_cycles = static_cast<unsigned long long>(layer_flops / (total_tiles * 1024.0));
    unsigned long long sim_cycles = 0;
    for (unsigned long long cycle = 0; cycle < base_compute_cycles; ++cycle) {
        sim_cycles++;
        // PMU Bank conflict hazard (mod 128)
        if (cycle % 128 == 0 || cycle % 128 == 32) {
            sim_cycles++;
        }
    }
    double simulated_compute_time_ms = static_cast<double>(sim_cycles) / 1e6;

    // Weight Load and NoC delays
    double weight_load_ms = weight_size_mb / (hbm_bandwidth * 1000.0 / 1024.0);
    
    unsigned int hops = static_cast<unsigned int>(grid_size * (2.0 / 3.0));
    double switch_ms = hops * 2e-6;
    double transfer_ms = compressed_act_size_mb / (noc_bandwidth * 1000.0 / 1024.0);
    double noc_routing_ms = switch_ms + transfer_ms;
    
    // MoE NoC router queue congestion delay
    if (wl.model_type == "moe") {
        noc_routing_ms += transfer_ms * 0.75;
    }

    // Prefetch overlap
    double prefetch_overlap = 0.94;
    double latency_ms = std::max(simulated_compute_time_ms, weight_load_ms) +
                        weight_load_ms * (1.0 - prefetch_overlap) +
                        noc_routing_ms + spill_time_ms;

    // Throughput and Economics
    double achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0);
    double utilization = (achieved_tflops / peak_compute_tflops) * 100.0;
    double tflops_per_dollar = achieved_tflops / silicon_cost;

    // Energy
    double dram_energy = ((weight_size_mb + spill_bytes_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 15e-12;
    double sram_energy = ((compressed_act_size_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 0.1e-12;
    double compute_energy = layer_flops * 0.025e-12;
    double total_energy = dram_energy + sram_energy + compute_energy;

    // Bottleneck
    std::string bottleneck = "Compute Pipeline Bound (Balanced Design)";
    if (spill_time_ms > 0.4 * latency_ms) {
        bottleneck = "Activation Spill Stall (SRAM Capacity Limit)";
    } else if (weight_load_ms > 1.2 * simulated_compute_time_ms) {
        bottleneck = "HBM Weight Streaming (Memory Bound)";
    } else if (wl.model_type == "moe" && (noc_routing_ms > 0.2 * latency_ms)) {
        bottleneck = "NoC Expert Routing Congestion";
    }

    std::stringstream s_lat, s_tflops, s_util, s_energy, s_cost_eff, s_spilled;
    s_lat << latency_ms;
    s_tflops << achieved_tflops;
    s_util << utilization;
    s_energy << total_energy;
    s_cost_eff << tflops_per_dollar;
    s_spilled << (spilled ? "Yes" : "No");

    result["arch_type"] = "RDU";
    result["model_name"] = wl.model_name;
    result["latency_ms"] = s_lat.str();
    result["achieved_tflops"] = s_tflops.str();
    result["pcu_utilization_pct"] = s_util.str();
    result["total_energy_j"] = s_energy.str();
    result["sram_spilled"] = s_spilled.str();
    result["tflops_per_dollar"] = s_cost_eff.str();
    result["bottleneck"] = bottleneck;

    return result;
}


// --- NPUSystemModel Implementation ---
NPUSystemModel::NPUSystemModel(unsigned int grid_size, double sram_capacity_mb, double hbm_bw, double bus_bw)
    : grid_size(grid_size), sram_capacity_mb(sram_capacity_mb), hbm_bandwidth(hbm_bw), bus_bandwidth(bus_bw) {
    total_pes = grid_size * grid_size;
    peak_compute_tflops = (total_pes * 2.0 * (1.0 * 1e9)) / 1e12; // 1GHz clock
    
    // NPU is physically compact (hardwired systolic, simple layout, high yield)
    // Sizing area and wafer cost per good die yields baseline silicon cost
    silicon_cost = (grid_size == 712) ? 22.01 : 22.52;
}

std::map<std::string, std::string> NPUSystemModel::simulate(const WorkloadSpec& wl) {
    std::map<std::string, std::string> result;

    double batch = wl.batch_size;
    double seq_len = wl.seq_len;
    double hidden_dim = wl.hidden_dim;
    double ffn_dim = wl.ffn_dim;
    double weight_size_mb = wl.weight_size_mb;

    // Estimate raw layer operations (scales with Batch Size)
    double layer_flops = 0.0;
    if (wl.model_type == "dense") {
        layer_flops = 2.0 * batch * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            2.0 * seq_len * seq_len * hidden_dim +
            seq_len * hidden_dim * hidden_dim +
            3.0 * seq_len * hidden_dim * ffn_dim
        );
    } else { // moe
        layer_flops = 2.0 * batch * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            seq_len * hidden_dim * ffn_dim * 2.0 * (wl.num_shared_experts + wl.routed_experts_per_token)
        );
    }
    double layer_gflops = layer_flops / 1e9;

    // Activation Sizing (scales with Batch Size, NO SRAM COMPRESSION SUPPORT!)
    double raw_act_size_mb = batch * (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0);

    // SRAM check (Raw central macro)
    bool spilled = (raw_act_size_mb > (sram_capacity_mb * 0.70));
    double spill_bytes_mb = 0.0;
    double spill_time_ms = 0.0;
    if (spilled) {
        spill_bytes_mb = raw_act_size_mb - (sram_capacity_mb * 0.70);
        spill_time_ms = (spill_bytes_mb * 2.0) / (hbm_bandwidth * 1000.0 / 1024.0);
    }

    // Systolic Shifting Cycle Loop
    unsigned long long base_compute_cycles = static_cast<unsigned long long>(layer_flops / (total_pes * 2.0));
    unsigned int setup_overhead = 2 * grid_size;
    unsigned long long sim_cycles = setup_overhead;

    for (unsigned long long cycle = 0; cycle < base_compute_cycles; ++cycle) {
        sim_cycles++;
        // Global bus contention (mod 32)
        if (cycle % 32 == 0 || cycle % 32 == 15) {
            sim_cycles += 2;
        }
    }
    double simulated_compute_time_ms = static_cast<double>(sim_cycles) / 1e6;

    // Memory Weight Thrashing (MoE weights loaded sequentially per token on NPU)
    double actual_weight_load_mb = weight_size_mb;
    if (wl.model_type == "moe") {
        double thrashing_factor = wl.routed_experts_per_token / 2.0; // Partial reuse
        actual_weight_load_mb = weight_size_mb * thrashing_factor;
    }
    double weight_load_ms = actual_weight_load_mb / (hbm_bandwidth * 1000.0 / 1024.0);

    // Global SRAM bus delays
    double sram_bus_delay_ms = (raw_act_size_mb + actual_weight_load_mb) / (bus_bandwidth * 1000.0 / 1024.0);

    // Shared-port central scratchpad limits prefetch to 10%
    double prefetch_overlap = 0.10;
    double latency_ms = simulated_compute_time_ms + 
                        weight_load_ms * (1.0 - prefetch_overlap) +
                        sram_bus_delay_ms + spill_time_ms;

    // Throughput and Economics
    double achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0);
    double utilization = (achieved_tflops / peak_compute_tflops) * 100.0;
    double tflops_per_dollar = achieved_tflops / silicon_cost;

    // Energy (Monolithic long buses charge at 0.5 pJ/bit vs RDU's 0.1 pJ/bit)
    double dram_energy = ((actual_weight_load_mb + spill_bytes_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 15e-12;
    double sram_energy = ((raw_act_size_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 0.5e-12;
    double compute_energy = layer_flops * 0.025e-12;
    double total_energy = dram_energy + sram_energy + compute_energy;

    // Bottleneck
    std::string bottleneck = "Compute Pipeline Bound (Balanced Design)";
    if (spill_time_ms > 0.4 * latency_ms) {
        bottleneck = "Activation Spill Stall (SRAM Capacity Limit)";
    } else if (weight_load_ms > 1.2 * simulated_compute_time_ms) {
        bottleneck = "HBM Weight Thrashing Memory Wall";
    } else if (sram_bus_delay_ms > 0.2 * latency_ms) {
        bottleneck = "Centralized Global Bus Routing Contention";
    }

    std::stringstream s_lat, s_tflops, s_util, s_energy, s_cost_eff, s_spilled;
    s_lat << latency_ms;
    s_tflops << achieved_tflops;
    s_util << utilization;
    s_energy << total_energy;
    s_cost_eff << tflops_per_dollar;
    s_spilled << (spilled ? "Yes" : "No");

    result["arch_type"] = "NPU";
    result["model_name"] = wl.model_name;
    result["latency_ms"] = s_lat.str();
    result["achieved_tflops"] = s_tflops.str();
    result["pcu_utilization_pct"] = s_util.str();
    result["total_energy_j"] = s_energy.str();
    result["sram_spilled"] = s_spilled.str();
    result["tflops_per_dollar"] = s_cost_eff.str();
    result["bottleneck"] = bottleneck;

    return result;
}
