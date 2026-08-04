#include "RDUMicroSim.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

// --- PCUPipeline Implementation ---
PCUPipeline::PCUPipeline() : stall_cycles(0), total_stalls(0) {
    for (int i = 0; i < 5; ++i) {
        current_stages[i] = static_cast<PipelineStage>(i);
    }
}

void PCUPipeline::tick() {
    if (stall_cycles > 0) {
        --stall_cycles;
        total_stalls++;
    } else {
        // Normal structural advancement of stages
        for (int i = 4; i > 0; --i) {
            current_stages[i] = current_stages[i - 1];
        }
        current_stages[0] = PipelineStage::FETCH;
    }
}

void PCUPipeline::stall(unsigned int cycles) {
    stall_cycles += cycles;
}


// --- PMUStorage Implementation ---
PMUStorage::PMUStorage(double capacity_kb, std::string compression)
    : capacity_kb(capacity_kb), compression_mode(compression) {}

double PMUStorage::get_compression_factor() const {
    if (compression_mode == "FP8") return 2.0;
    if (compression_mode == "INT4") return 4.0;
    return 1.0;
}

bool PMUStorage::check_and_allocate_activations(double activation_size_mb, double aggregate_capacity_mb) {
    double compressed_size = activation_size_mb / get_compression_factor();
    double sram_budget_mb = aggregate_capacity_mb * 0.70; // 70% of total memory is assigned for active PMU buffers
    return (compressed_size <= sram_budget_mb);
}

bool PMUStorage::has_bank_conflict(unsigned int cycle) const {
    // Structural cycle-approximate model for bank conflicts:
    // If the clock cycle mod 128 falls on specific boundaries,
    // we simulate dual-port read/write structural hazards on the local 8T cells.
    return (cycle % 128 == 0 || cycle % 128 == 32);
}


// --- NoCMeshRouter Implementation ---
NoCMeshRouter::NoCMeshRouter(double link_bw_gb_s, unsigned int grid_size)
    : link_bandwidth(link_bw_gb_s), grid_size(grid_size) {}

double NoCMeshRouter::model_routing_delay(double data_size_mb, unsigned int manhattan_hops, bool is_moe) const {
    // 2 ns switch delay per hop @ 1GHz
    double switch_delay_ms = manhattan_hops * 2e-6;
    double transfer_ms = data_size_mb / (link_bandwidth * 1000.0 / 1024.0);
    
    double congestion_factor = 1.0;
    if (is_moe) {
        // Many-to-one token routed expert hot-spots create credit-based flow backpressure.
        // We model congestion scaling up by density.
        congestion_factor = 1.75;
    }
    
    return switch_delay_ms + (transfer_ms * congestion_factor);
}


// --- HBMInterface Implementation ---
HBMInterface::HBMInterface(double bandwidth_gb_s) : bandwidth(bandwidth_gb_s) {}

double HBMInterface::model_weight_load_delay(double weight_size_mb) const {
    return weight_size_mb / (bandwidth * 1000.0 / 1024.0);
}


// --- RDUSystem Implementation ---
RDUSystem::RDUSystem(unsigned int grid_size, double sram_per_pmu_kb, std::string comp_mode, double hbm_bw, double noc_bw)
    : grid_size(grid_size), sram_per_pmu_kb(sram_per_pmu_kb), compression_mode(comp_mode),
      hbm_bandwidth(hbm_bw), noc_bandwidth(noc_bw) {
    total_tiles = grid_size * grid_size;
    peak_compute_tflops = (total_tiles * 1024.0 * (1.0 * 1e9)) / 1e12; // 1GHz base clock
    total_sram_mb = (total_tiles * sram_per_pmu_kb) / 1024.0;
}

std::map<std::string, std::string> RDUSystem::simulate_layer(const ModelSpec& spec) {
    std::map<std::string, std::string> result;

    // 1. Structural Parameters
    double seq_len = spec.seq_len;
    double hidden_dim = spec.hidden_dim;
    double ffn_dim = spec.ffn_dim;
    double weight_size_mb = spec.weight_size_mb;

    // Estimate raw layer operations (GFLOPs)
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
            3.0 * seq_len * hidden_dim * hidden_dim + // attention
            seq_len * hidden_dim * ffn_dim * 2.0 * (spec.num_shared_experts + spec.routed_experts_per_token)
        );
    }
    double layer_gflops = layer_flops / 1e9;

    // 2. Instantiate structural simulation pipelines
    PCUPipeline pipeline;
    PMUStorage pmu(sram_per_pmu_kb, compression_mode);
    NoCMeshRouter router(noc_bandwidth, grid_size);
    HBMInterface hbm(hbm_bandwidth);

    // Calculate activation sizing
    double raw_act_size_mb = (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0);
    double comp_factor = pmu.get_compression_factor();
    double compressed_act_size_mb = raw_act_size_mb / comp_factor;

    // Check for SRAM spilling
    bool spilled = !pmu.check_and_allocate_activations(raw_act_size_mb, total_sram_mb);
    double spill_bytes_mb = 0.0;
    double spill_time_ms = 0.0;
    if (spilled) {
        spill_bytes_mb = compressed_act_size_mb - (total_sram_mb * 0.70);
        spill_time_ms = (spill_bytes_mb * 2.0) / (hbm_bandwidth * 1000.0 / 1024.0);
    }

    // 3. Cycle-Approximate Simulation loop
    // Convert mathematical workloads to cycle counts:
    // Raw base compute cycles at perfect pipeline concurrency = FLOPs / (total_tiles * 1024 MAC_ops * 2)
    unsigned long long base_compute_cycles = static_cast<unsigned long long>(layer_flops / (total_tiles * 1024.0));
    
    // Simulate pipeline transitions and structural hazards cycle-by-cycle
    unsigned long long sim_cycles = 0;
    for (unsigned long long cycle = 0; cycle < base_compute_cycles; ++cycle) {
        pipeline.tick();
        sim_cycles++;
        
        // Inject bank-conflict stalls
        if (pmu.has_bank_conflict(cycle)) {
            pipeline.stall(1); // 1 stall cycle injected
            sim_cycles++;
        }
    }

    double simulated_compute_time_ms = static_cast<double>(sim_cycles) / 1e6; // @ 1GHz

    // 4. Memory Stream and NoC Transit Delays
    double weight_load_ms = hbm.model_weight_load_delay(weight_size_mb);
    
    // Average Manhattan distance on mesh
    unsigned int manhattan_hops = static_cast<unsigned int>(grid_size * (2.0 / 3.0));
    bool is_moe = (spec.model_type == "moe");
    double noc_routing_ms = router.model_routing_delay(compressed_act_size_mb, manhattan_hops, is_moe);

    // PMU double-buffering pre-fetch overlap factor
    double prefetch_overlap = (sram_per_pmu_kb == 128) ? 0.94 : 0.97;

    // Latency = Max(Compute, Weights) + Non-overlapped Weight Stream + NoC Routing + Spills
    double total_latency_ms = std::max(simulated_compute_time_ms, weight_load_ms) +
                              weight_load_ms * (1.0 - prefetch_overlap) +
                              noc_routing_ms + spill_time_ms;

    // Achieved metrics
    double achieved_tflops = (layer_gflops / 1000.0) / (total_latency_ms / 1000.0);
    double utilization = (achieved_tflops / peak_compute_tflops) * 100.0;

    // Energy metrics
    double dram_energy = ((weight_size_mb + spill_bytes_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 15e-12;
    double sram_energy = ((compressed_act_size_mb * 2.0) * 1024.0 * 1024.0 * 8.0) * 0.1e-12;
    double compute_energy = layer_flops * 0.025e-12;
    double total_energy = dram_energy + sram_energy + compute_energy;

    // Identify primary bottleneck
    std::string bottleneck = "Compute Pipeline Bound (Balanced Design)";
    if (spill_time_ms > 0.4 * total_latency_ms) {
        bottleneck = "Activation Spill Stall (SRAM Capacity Limit)";
    } else if (weight_load_ms > 1.2 * simulated_compute_time_ms) {
        bottleneck = "HBM Weight Streaming (Memory Bound)";
    } else if (is_moe && (noc_routing_ms > 0.2 * total_latency_ms)) {
        bottleneck = "NoC Expert Routing Congestion (NoC Bandwidth Limit)";
    }

    // Format output strings
    std::stringstream s_lat, s_tflops, s_util, s_energy, s_tiles, s_sram, s_peak, s_stalls, s_spilled;
    s_lat << total_latency_ms;
    s_tflops << achieved_tflops;
    s_util << utilization;
    s_energy << total_energy;
    s_tiles << total_tiles;
    s_sram << total_sram_mb;
    s_peak << peak_compute_tflops;
    s_stalls << pipeline.get_total_stalls();
    s_spilled << (spilled ? "Yes" : "No");

    result["model_name"] = spec.model_name;
    result["model_type"] = spec.model_type;
    result["total_tiles"] = s_tiles.str();
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
