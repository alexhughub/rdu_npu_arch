#!/usr/bin/env python3
"""
RDU Microarchitectural and Pipeline-Approximate Core Simulator.
Models 2D mesh, NoC routing, PMU SRAM ring-buffering, HBM bandwidth,
activation compression, and dynamic Mixture-of-Experts (MoE) token routing.
"""

import os
import configparser
import json
import numpy as np

class RDUMicroSim:
    def __init__(self, config_path=None):
        self.grid_size = 32          # 32x32 = 1024 tiles
        self.sram_per_pmu_kb = 128   # KB
        self.pcu_frequency_ghz = 1.0 # GHz
        self.hbm_bandwidth_gb_s = 2400.0 # GB/s
        self.noc_link_bandwidth_gb_s = 256.0 # GB/s
        self.compression_mode = 'INT4' # None, FP8, INT4
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
        self.total_tiles = self.grid_size * self.grid_size
        # Each tile contains 1 PCU (housing BF16 Tensor Engine executing 512 MACs per cycle)
        # BF16 Tensor Engine FLOPS per cycle = 512 MACs * 2 = 1024 FLOPs
        # Peak FLOPs per tile at 1GHz = 1.024 TFLOPS.
        self.peak_compute_tflops = (self.total_tiles * 1024.0 * (self.pcu_frequency_ghz * 1e9)) / 1e12
        self.total_sram_mb = (self.total_tiles * self.sram_per_pmu_kb) / 1024.0

    def load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'RDU_Hardware_Spec' in config:
            spec = config['RDU_Hardware_Spec']
            self.grid_size = int(spec.get('grid_size', '32'))
            self.sram_per_pmu_kb = int(spec.get('sram_per_pmu_kb', '128'))
            self.pcu_frequency_ghz = float(spec.get('pcu_frequency_ghz', '1.0'))
            self.hbm_bandwidth_gb_s = float(spec.get('hbm_bandwidth_gb_s', '2400.0'))
            self.noc_link_bandwidth_gb_s = float(spec.get('noc_link_bandwidth_gb_s', '256.0'))
            self.compression_mode = spec.get('compression_mode', 'INT4')

    def simulate_layer(self, model_spec):
        """
        Simulates running a single layer of an LLM.
        Returns detailed latency (ms), energy (Joules), throughput (TFLOPS), and bottleneck analysis.
        """
        model_name = model_spec['model_name']
        model_type = model_spec['type']
        seq_len = model_spec['seq_len']
        weight_size_mb = model_spec['weight_size_mb']
        hidden_dim = model_spec.get('hidden_dim', 8192)
        ffn_dim = model_spec.get('ffn_dim', 28672)
        
        # 1. Activation Sizing & Compression Modeling
        # Active activation size in MB (FP16 base representation)
        raw_act_size_mb = (seq_len * hidden_dim * 2 * 4) / (1024.0 * 1024.0) # approx layer activations
        
        comp_factor = 1.0
        if self.compression_mode == 'FP8':
            comp_factor = 2.0
        elif self.compression_mode == 'INT4':
            comp_factor = 4.0
            
        compressed_act_size_mb = raw_act_size_mb / comp_factor
        
        # Check if activations exceed SRAM capacity.
        # RDU PMUs store weights (pre-fetched slice) and activations.
        # Weights are double-buffered. Activations reside on-chip in a Ring Buffer.
        # Active SRAM budget available for activations ~ 70% of total SRAM.
        active_sram_mb = self.total_sram_mb * 0.70
        
        act_spill_bytes_mb = 0.0
        act_spill_time_ms = 0.0
        sram_spilled = False
        
        if compressed_act_size_mb > active_sram_mb:
            sram_spilled = True
            act_spill_bytes_mb = compressed_act_size_mb - active_sram_mb
            # Activation spill requires writing to HBM and reading back
            act_spill_time_ms = (act_spill_bytes_mb * 2.0) / (self.hbm_bandwidth_gb_s * 1000.0 / 1024.0)
            
        # 2. Compute Latency Modeling
        # Estimate total layer operations (GFLOPs)
        if model_type == 'dense':
            layer_flops = 2 * (
                3 * seq_len * hidden_dim**2 +
                2 * seq_len**2 * hidden_dim +
                seq_len * hidden_dim**2 +
                3 * seq_len * hidden_dim * ffn_dim
            )
        else: # moe layer (DeepSeek)
            # Shared expert + routed experts calculations
            layer_flops = 2 * (
                3 * seq_len * hidden_dim**2 + # attention
                seq_len * hidden_dim * ffn_dim * 2 * (model_spec.get('num_shared_experts', 1) + model_spec.get('routed_experts_per_token', 8))
            )
            
        layer_gflops = layer_flops / 1e9
        
        # Raw Compute Execution Time (PCU grid utilization at 92%)
        pcu_util = 0.92
        compute_time_ms = (layer_gflops / (self.peak_compute_tflops * pcu_util))
        
        # 3. Memory & Streaming Latency (HBM to PMU)
        # DRAM weight fetch time
        weight_stream_time_ms = weight_size_mb / (self.hbm_bandwidth_gb_s * 1000.0 / 1024.0)
        
        # 4. NoC Routing Latency and Congestion (Detailed modeling based on topology)
        # NoC Manhattan Switch Delay: 2 ns per hop @ 1GHz
        switch_delay_ms = 2e-6
        avg_manhattan_hops = self.grid_size * (2.0 / 3.0) # avg distance on 2D mesh
        
        noc_transit_base_ms = avg_manhattan_hops * switch_delay_ms
        noc_transfer_bandwidth_ms = (compressed_act_size_mb) / (self.noc_link_bandwidth_gb_s * 1000.0 / 1024.0)
        noc_total_latency_ms = noc_transit_base_ms + noc_transfer_bandwidth_ms
        
        # MoE Dynamic Routing Congestion model
        moe_congestion_overhead_ms = 0.0
        if model_type == 'moe':
            # Routed experts are mapped across the 1024 tile grid.
            # When tokens route to active experts, it creates a NoC hot-spot (many-to-one routing).
            # We model congestion based on: (routed experts per token) / (grid dimensions ratio)
            active_expert_density = model_spec.get('routed_experts_per_token', 8) / float(self.grid_size)
            congestion_factor = max(1.0, active_expert_density * 3.5)
            # Congestion increases NoC transfer time
            moe_congestion_overhead_ms = noc_transfer_bandwidth_ms * (congestion_factor - 1.0)
            noc_total_latency_ms += moe_congestion_overhead_ms
            
        # 5. Pipeline Overlap Modeling
        # Ring-buffering enables pre-fetch overlap of weights.
        pmu_overlap_factor = 0.94 if self.sram_per_pmu_kb == 128 else 0.97
        
        # Latency: Max(Compute, Weights) + Non-overlapped Weights + NoC transit + Activation Spills
        latency_ms = max(compute_time_ms, weight_stream_time_ms) + weight_stream_time_ms * (1.0 - pmu_overlap_factor) + noc_total_latency_ms + act_spill_time_ms
        
        # Achieved Performance
        achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0)
        pcu_utilization_pct = (achieved_tflops / self.peak_compute_tflops) * 100.0
        
        # 6. Active Energy Modeling (Joules)
        # DRAM access cost: 15 pJ/bit
        dram_energy_j = ((weight_size_mb + act_spill_bytes_mb * 2) * 1024 * 1024 * 8) * 15e-12
        # SRAM access cost: 0.1 pJ/bit (compressed on-chip representation)
        sram_energy_j = ((compressed_act_size_mb * 2) * 1024 * 1024 * 8) * 0.1e-12
        # Compute dynamic cost: 0.05 pJ per MAC (0.05 pJ / 2 FLOPs = 0.025 pJ/FLOP)
        compute_energy_j = (layer_flops) * 0.025e-12
        
        total_energy_j = dram_energy_j + sram_energy_j + compute_energy_j
        
        # Bottleneck Identification
        if act_spill_time_ms > 0.4 * latency_ms:
            bottleneck = "Activation Spill Stall (SRAM Capacity Limit)"
        elif weight_stream_time_ms > 1.2 * compute_time_ms:
            bottleneck = "HBM Weight Streaming (Memory Bound)"
        elif moe_congestion_overhead_ms > 0.2 * latency_ms:
            bottleneck = "NoC Expert Routing Congestion (NoC Bandwidth Limit)"
        else:
            bottleneck = "Compute Pipeline Bound (Balanced Design)"
            
        return {
            'model_name': model_name,
            'model_type': model_type,
            'total_tiles': self.total_tiles,
            'total_sram_mb': self.total_sram_mb,
            'peak_compute_tflops': self.peak_compute_tflops,
            'latency_ms': latency_ms,
            'compute_time_ms': compute_time_ms,
            'weight_stream_time_ms': weight_stream_time_ms,
            'noc_total_latency_ms': noc_total_latency_ms,
            'act_spill_time_ms': act_spill_time_ms,
            'achieved_tflops': achieved_tflops,
            'pcu_utilization_pct': pcu_utilization_pct,
            'total_energy_j': total_energy_j,
            'sram_spilled': sram_spilled,
            'bottleneck': bottleneck
        }

if __name__ == '__main__':
    # Direct test
    sim = RDUMicroSim()
    llama = {
        'model_name': 'LLaMA-3-70B',
        'type': 'dense',
        'seq_len': 8192,
        'weight_size_mb': 1856.0,
        'hidden_dim': 8192,
        'ffn_dim': 28672
    }
    res = sim.simulate_layer(llama)
    print(f"Test Run LLaMA-3-70B: {res['achieved_tflops']:.2f} TFLOPS | Bottleneck: {res['bottleneck']}")
