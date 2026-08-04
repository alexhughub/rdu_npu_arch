#ifndef RDU_MICRO_SIM_HPP
#define RDU_MICRO_SIM_HPP

#include <string>
#include <vector>
#include <map>

// Target LLM workload specifications
struct ModelSpec {
    std::string model_name;
    std::string model_type; // "dense" or "moe"
    double seq_len;
    double weight_size_mb;
    double hidden_dim;
    double ffn_dim;
    double num_shared_experts;
    double num_routed_experts;
    double routed_experts_per_token;
};

// Simulated pipeline stages in each PCU SIMD vector lane
enum class PipelineStage {
    FETCH = 0,
    DECODE,
    REG_READ,
    EXECUTE, // Tensor MAC Engine
    WRITEBACK,
    NUM_STAGES
};

// Structural model of the PCU vector pipeline
class PCUPipeline {
public:
    PCUPipeline();
    void tick(); // Progresses pipeline by 1 cycle
    void stall(unsigned int cycles);
    bool is_stalled() const { return stall_cycles > 0; }
    unsigned int get_total_stalls() const { return total_stalls; }

private:
    unsigned int stall_cycles;
    unsigned int total_stalls;
    PipelineStage current_stages[5];
};

// Structural model of the PMU Storage (Distributed SRAM cells)
class PMUStorage {
public:
    PMUStorage(double capacity_kb, std::string compression);
    bool check_and_allocate_activations(double activation_size_mb, double aggregate_capacity_mb);
    bool has_bank_conflict(unsigned int cycle) const; // Models bank conflict cycles
    double get_compression_factor() const;

private:
    double capacity_kb;
    std::string compression_mode;
};

// Structural model of the 2D NoC routing routers (mesh switch buffers)
class NoCMeshRouter {
public:
    NoCMeshRouter(double link_bw_gb_s, unsigned int grid_size);
    double model_routing_delay(double data_size_mb, unsigned int manhattan_hops, bool is_moe) const;

private:
    double link_bandwidth;
    unsigned int grid_size;
};

// Structural model of the High-Bandwidth Memory controller and FIFO queues
class HBMInterface {
public:
    HBMInterface(double bandwidth_gb_s);
    double model_weight_load_delay(double weight_size_mb) const;

private:
    double bandwidth;
};

// Complete RDU System orchestrator
class RDUSystem {
public:
    RDUSystem(unsigned int grid_size, double sram_per_pmu_kb, std::string comp_mode, double hbm_bw, double noc_bw);
    
    // Core cycle-approximate execution loop for a single layer
    std::map<std::string, std::string> simulate_layer(const ModelSpec& spec);

private:
    unsigned int grid_size;
    double sram_per_pmu_kb;
    std::string compression_mode;
    double hbm_bandwidth;
    double noc_bandwidth;

    unsigned int total_tiles;
    double peak_compute_tflops;
    double total_sram_mb;
};

#endif // RDU_MICRO_SIM_HPP
