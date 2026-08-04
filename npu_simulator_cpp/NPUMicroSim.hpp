#ifndef NPU_MICRO_SIM_HPP
#define NPU_MICRO_SIM_HPP

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

// Structural model of a single Processing Element (PE) inside the Systolic Grid
class SystolicPE {
public:
    SystolicPE();
    void tick(); // Progresses PE state by 1 cycle (shifts registers)
    void reset();
    
private:
    double weight_reg;
    double input_reg;
    double accumulator_reg;
};

// Structural model of the Monolithic Centralized SRAM Block
class CentralSRAM {
public:
    CentralSRAM(double capacity_mb);
    bool check_and_allocate(double data_size_mb);
    double get_capacity() const { return capacity_mb; }

private:
    double capacity_mb;
};

// Structural model of the Central SRAM Global Bus & Shared Port Controller
class GlobalBus {
public:
    GlobalBus(double bandwidth_gb_s);
    bool has_port_contention(unsigned int cycle) const; // Models shared port port contention
    double get_bandwidth() const { return bandwidth; }

private:
    double bandwidth;
};

// Structural model of the HBM memory controller
class HBMInterface {
public:
    HBMInterface(double bandwidth_gb_s);
    double get_bandwidth() const { return bandwidth; }

private:
    double bandwidth;
};

// Unified Centralized NPU System Simulator
class NPUSystem {
public:
    NPUSystem(unsigned int grid_size, double sram_mb, double hbm_bw, double bus_bw);
    std::map<std::string, std::string> simulate_layer(const ModelSpec& spec);

private:
    unsigned int grid_size;
    double sram_capacity_mb;
    double hbm_bandwidth;
    double bus_bandwidth;

    unsigned int total_pes;
    double peak_compute_tflops;
};

#endif // NPU_MICRO_SIM_HPP
