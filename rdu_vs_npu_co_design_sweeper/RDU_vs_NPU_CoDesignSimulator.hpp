#ifndef RDU_VS_NPU_CO_DESIGN_SIM_HPP
#define RDU_VS_NPU_CO_DESIGN_SIM_HPP

#include <string>
#include <vector>
#include <map>

// Unified workload specifications with Batch Size and Context Sequence Length
struct WorkloadSpec {
    std::string model_name;
    std::string model_type; // "dense" or "moe"
    double batch_size;       // Batch size (e.g. 1 or 128)
    double seq_len;          // Sequence context length (e.g. 512 or 32768)
    double weight_size_mb;   // Base layers weight size
    double hidden_dim;
    double ffn_dim;
    double num_shared_experts;
    double num_routed_experts;
    double routed_experts_per_token;
};

// Structural RDU model
class RDUSystemModel {
public:
    RDUSystemModel(unsigned int grid_size, double sram_per_pmu_kb, std::string comp_mode, double hbm_bw, double noc_bw);
    std::map<std::string, std::string> simulate(const WorkloadSpec& wl);

private:
    unsigned int grid_size;
    double sram_per_pmu_kb;
    std::string compression_mode;
    double hbm_bandwidth;
    double noc_bandwidth;

    unsigned int total_tiles;
    double peak_compute_tflops;
    double total_sram_mb;
    double silicon_cost;
};

// Structural Centralized NPU model
class NPUSystemModel {
public:
    NPUSystemModel(unsigned int grid_size, double sram_capacity_mb, double hbm_bw, double bus_bw);
    std::map<std::string, std::string> simulate(const WorkloadSpec& wl);

private:
    unsigned int grid_size;
    double sram_capacity_mb;
    double hbm_bandwidth;
    double bus_bandwidth;

    unsigned int total_pes;
    double peak_compute_tflops;
    double silicon_cost;
};

#endif // RDU_VS_NPU_CO_DESIGN_SIM_HPP
