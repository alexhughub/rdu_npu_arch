#include "NPUMicroSim.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

// --- SystolicPE Implementation ---
SystolicPE::SystolicPE() : weight_reg(0.0), input_reg(0.0), accumulator_reg(0.0) {}

void SystolicPE::tick() {
    // Models cycle-by-cycle data shifting (weight-stationary systolic array)
    // Data shifts from West to East registers
    accumulator_reg += input_reg * weight_reg;
}

void SystolicPE::reset() {
    weight_reg = 0.0;
    input_reg = 0.0;
    accumulator_reg = 0.0;
}


// --- CentralSRAM Implementation ---
CentralSRAM::CentralSRAM(double capacity_mb) : capacity_mb(capacity_mb) {}

bool CentralSRAM::check_and_allocate(double data_size_mb) {
    // 70% of total centralized scratchpad SRAM is reserved for active layer buffers
    double active_sram_mb = capacity_mb * 0.70;
    return (data_size_mb <= active_sram_mb);
}


// --- GlobalBus Implementation ---
GlobalBus::GlobalBus(double bandwidth_gb_s) : bandwidth(bandwidth_gb_s) {}

bool GlobalBus::has_port_contention(unsigned int cycle) const {
    // Centralized memory scratchpad has shared read/write ports.
    // If multiple PE rows attempt load/write actions on the same clock cycle,
    // we simulate shared port arbitration conflicts.
    return (cycle % 32 == 0 || cycle % 32 == 15);
}


// --- HBMInterface Implementation ---
HBMInterface::HBMInterface(double bandwidth_gb_s) : bandwidth(bandwidth_gb_s) {}


// --- NPUSystem Implementation ---
NPUSystem::NPUSystem(unsigned int grid_size, double sram_mb, double hbm_bw, double bus_bw)
    : grid_size(grid_size), sram_capacity_mb(sram_mb), hbm_bandwidth(hbm_bw), bus_bandwidth(bus_bw) {
    total_pes = grid_size * grid_size;
    peak_compute_tflops = (total_pes * 2.0 * (1.0 * 1e9)) / 1e12; // 1GHz base clock
}

std::map<std::string, std::string> NPUSystem::simulate_layer(const ModelSpec& spec) {
    std::map<std::string, std::string> result;

    double seq_len = spec.seq_len;
    double hidden_dim = spec.hidden_dim;
    double ffn_dim = spec.ffn_dim;
    double weight_size_mb = spec.weight_size_mb;

    // 1. Estimate raw operations (GFLOPs)
    double layer_flops = 0.0;
    if (spec.model_type == "dense") {
        layer_flops = 2.0 * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            2.0 * seq_len * seq_len * hidden_dim +
            seq_len * hidden_dim * hidden_dim +
            3.0 * seq_len * hidden_dim * ffn_dim
        );
    } else { // moe
        layer_flops = 2.0 * (
            3.0 * seq_len * hidden_dim * hidden_dim +
            seq_len * hidden_dim * ffn_dim * 2.0 * (spec.num_shared_experts + spec.routed_experts_per_token)
        );
    }
    double layer_gflops = layer_flops / 1e9;

    // 2. Sizing and spilling (Zero compression supported inside centralized SRAM block)
    double raw_act_size_mb = (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0);
    CentralSRAM central_sram(sram_capacity_mb);
    bool spilled = !central_sram.check_and_allocate(raw_act_size_mb);
    
    double spill_bytes_mb = 0.0;
    double spill_time_ms = 0.0;
    if (spilled) {
        spill_bytes_mb = raw_act_size_mb - (sram_capacity_mb * 0.70);
        spill_time_ms = (spill_bytes_mb * 2.0) / (hbm_bandwidth * 1000.0 / 1024.0);
    }

    // 3. Cycle-Approximate Systolic Shifting Simulation
    // In systolic arrays, data must shift row-by-row and col-by-col.
    // The base compute cycles of a 2D matrix multiplication on grid size N:
    // Base cycles = FLOPs / (N * N * 2).
    // Plus, we model the systolic setup phase (filling row/col pipeline over 2*N cycles)
    unsigned long long base_compute_cycles = static_cast<unsigned long long>(layer_flops / (total_pes * 2.0));
    unsigned int setup_overhead_cycles = 2 * grid_size;
    
    unsigned long long sim_cycles = setup_overhead_cycles;
    GlobalBus global_bus(bus_bandwidth);

    for (unsigned long long cycle = 0; cycle < base_compute_cycles; ++cycle) {
        sim_cycles++;
        // Inject global bus shared port contention stalls
        if (global_bus.has_port_contention(cycle)) {
            sim_cycles += 2; // 2 stall cycles injected due to global bus contention
        }
    }

    double simulated_compute_time_ms = static_cast<double>(sim_cycles) / 1e6; // @ 1GHz

    // 4. Memory Stream and Weight Thrashing (MoE only)
    // Under MoE, we cannot map experts spatially because PEs are temporally mapped.
    // Therefore, expert weights must be loaded dynamically over HBM.
    double actual_weight_load_mb = weight_size_mb;
    if (spec.model_type == "moe") {
        double thrashing_factor = spec.routed_experts_per_token / 2.0; // Partial weight reuse on batch
        actual_weight_load_mb = weight_size_mb * thrashing_factor;
    }
    
    double weight_load_ms = actual_weight_load_mb / (hbm_bandwidth * 1000.0 / 1024.0);

    // Global SRAM Bus read/write delay
    double sram_bus_delay_ms = (raw_act_size_mb + actual_weight_load_mb) / (bus_bandwidth * 1000.0 / 1024.0);

    // Centralized memory scratchpad has shared ports, limiting prefetch to 10%
    double prefetch_overlap = 0.10;

    // Latency = Compute + Non-overlapped Weight stream + Bus contention + Spills
    double total_latency_ms = simulated_compute_time_ms + 
                              weight_load_ms * (1.0 - prefetch_overlap) +
                              sram_bus_delay_ms + spill_time_ms;

    // Achieved metrics
    double achieved_tflops = (layer_gflops / 1000.0) / (total_latency_ms / 1000.0);
    double utilization = (achieved_tflops / peak_compute_tflops) * 100.0;

    // Energy metrics
    double dram_energy = ((actual_weight_load_mb + spill_bytes_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 15e-12;
    // High wire charging capacitance on centralized bus (0.5 pJ/bit vs. RDU's 0.1 pJ/bit)
    double sram_energy = ((raw_act_size_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 0.5e-12;
    double compute_energy = layer_flops * 0.025e-12;
    double total_energy = dram_energy + sram_energy + compute_energy;

    // Identify primary bottleneck
    std::string bottleneck = "Compute Pipeline Bound (Balanced Design)";
    if (spill_time_ms > 0.4 * total_latency_ms) {
        bottleneck = "Activation Spill Stall (SRAM Capacity Limit)";
    } else if (weight_load_ms > 1.2 * simulated_compute_time_ms) {
        bottleneck = "HBM Weight Thrashing Memory Wall";
    } else if (sram_bus_delay_ms > 0.2 * total_latency_ms) {
        bottleneck = "Centralized Global Bus Routing Contention";
    }

    // Format output strings
    std::stringstream s_lat, s_tflops, s_util, s_energy, s_pes, s_sram, s_peak, s_stalls, s_spilled;
    s_lat << total_latency_ms;
    s_tflops << achieved_tflops;
    s_util << utilization;
    s_energy << total_energy;
    s_pes << total_pes;
    s_sram << sram_capacity_mb;
    s_peak << peak_compute_tflops;
    s_stalls << (sim_cycles - base_compute_cycles);
    s_spilled << (spilled ? "Yes" : "No");

    result["model_name"] = spec.model_name;
    result["model_type"] = spec.model_type;
    result["total_tiles"] = s_pes.str();
    result["total_sram_mb"] = s_sram.str();
    result["peak_compute_tflops"] = s_peak.str();
    result["latency_ms"] = s_lat.str();
    result["achieved_tflops"] = s_tflops.str();
    result["pcu_utilization_pct"] = s_util.str();
    result["total_energy_j"] = s_energy.str();
    result["sram_spilled"] = s_spilled.str();
    result["pipeline_stalls"] = s_stalls.str();
    result["bottleneck"] = bottleneck;

    return result;
}
