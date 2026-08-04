#!/usr/bin/env python3
"""
TPU-style Centralized Systolic NPU Microarchitectural Core Simulator.
Models monolithic PE array computation, shared central SRAM global bus congestion,
zero-compression activation spilling, and MoE expert weight thrashing.
"""

import os
import configparser
import json
import numpy as np

class NPUMicroSim:
    def __init__(self, config_path=None):
        self.grid_size = 512            # 512x512 PE Array = 262,144 MACs
        self.sram_capacity_mb = 128.0   # MB centralized SRAM
        self.frequency_ghz = 1.0        # GHz
        self.hbm_bandwidth_gb_s = 2400.0 # GB/s
        self.sram_bus_bandwidth_gb_s = 4800.0 # GB/s
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
        self.total_pes = self.grid_size * self.grid_size
        # Peak FLOPs = total_pes * 2 FLOPs/cycle * clock frequency
        self.peak_compute_tflops = (self.total_pes * 2.0 * (self.frequency_ghz * 1e9)) / 1e12

    def load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'NPU_Hardware_Spec' in config:
            spec = config['NPU_Hardware_Spec']
            self.grid_size = int(spec.get('grid_size', '512'))
            self.sram_capacity_mb = float(spec.get('sram_capacity_mb', '128.0'))
            self.frequency_ghz = float(spec.get('frequency_ghz', '1.0'))
            self.hbm_bandwidth_gb_s = float(spec.get('hbm_bandwidth_gb_s', '2400.0'))
            self.sram_bus_bandwidth_gb_s = float(spec.get('sram_bus_bandwidth_gb_s', '4800.0'))

    def simulate_layer(self, model_spec):
        """
        Simulates running a single layer of an LLM.
        Returns latency (ms), energy (Joules), throughput (TFLOPS), and bottleneck analysis.
        """
        model_name = model_spec['model_name']
        model_type = model_spec['type']
        seq_len = model_spec['seq_len']
        weight_size_mb = model_spec['weight_size_mb']
        hidden_dim = model_spec.get('hidden_dim', 8192)
        ffn_dim = model_spec.get('ffn_dim', 28672)
        
        # 1. Activation Sizing (No SRAM Compression Support in raw Centralized block)
        raw_act_size_mb = (seq_len * hidden_dim * 2 * 4) / (1024.0 * 1024.0)
        
        # Centralized SRAM must store both weights (active double buffer) and activations.
        # Since SRAM is raw memory macro, we cannot compress activations inside the SRAM Columns.
        active_sram_mb = self.sram_capacity_mb * 0.70
        
        act_spill_bytes_mb = 0.0
        act_spill_time_ms = 0.0
        sram_spilled = False
        
        if raw_act_size_mb > active_sram_mb:
            sram_spilled = True
            act_spill_bytes_mb = raw_act_size_mb - active_sram_mb
            # Spill reads and writes activations off-chip to HBM
            act_spill_time_ms = (act_spill_bytes_mb * 2.0) / (self.hbm_bandwidth_gb_s * 1000.0 / 1024.0)
            
        # 2. Compute Latency Modeling
        # Estimate total operations (GFLOPs)
        if model_type == 'dense':
            layer_flops = 2 * (
                3 * seq_len * hidden_dim**2 +
                2 * seq_len**2 * hidden_dim +
                seq_len * hidden_dim**2 +
                3 * seq_len * hidden_dim * ffn_dim
            )
        else: # moe layer
            layer_flops = 2 * (
                3 * seq_len * hidden_dim**2 + # attention
                seq_len * hidden_dim * ffn_dim * 2 * (model_spec.get('num_shared_experts', 1) + model_spec.get('routed_experts_per_token', 8))
            )
        layer_gflops = layer_flops / 1e9
        
        # Systolic Array execution efficiency is high on dense (96%), but lower on MoE due to padding
        pcu_util = 0.96 if model_type == 'dense' else 0.72
        compute_time_ms = (layer_gflops / (self.peak_compute_tflops * pcu_util))
        
        # 3. Memory & Streaming Latency (HBM weight fetches)
        # In Mixture of Experts (MoE), systolic arrays suffer from Weight Thrashing!
        # Because we cannot route tokens spatially on-chip to experts, we must fetch 
        # expert weights dynamically from HBM as tokens stream through.
        # Weight load volume scales up by active routed experts per token.
        actual_weight_load_mb = weight_size_mb
        if model_type == 'moe':
            thrashing_factor = model_spec.get('routed_experts_per_token', 8) / 2.0 # Assume some basic temporal reuse
            actual_weight_load_mb = weight_size_mb * thrashing_factor
            
        weight_stream_time_ms = actual_weight_load_mb / (self.hbm_bandwidth_gb_s * 1000.0 / 1024.0)
        
        # 4. Global SRAM Bus Contention Delays
        # All PEs must load inputs/write outputs to Central SRAM over a shared global bus.
        sram_bus_delay_ms = (raw_act_size_mb + actual_weight_load_mb) / (self.sram_bus_bandwidth_gb_s * 1000.0 / 1024.0)
        
        # 5. Overlap modeling
        # Centralized SRAM port sharing limits prefetch.
        # Compute and weight prefetch contend for the same shared scratchpad ports.
        npu_overlap_factor = 0.10 # Max 10% overlap
        
        # Latency = Compute + Non-overlapped Weight stream + Bus contention + Activation Spills
        latency_ms = compute_time_ms + weight_stream_time_ms * (1.0 - npu_overlap_factor) + sram_bus_delay_ms + act_spill_time_ms
        
        # Achieved performance
        achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0)
        pcu_utilization_pct = (achieved_tflops / self.peak_compute_tflops) * 100.0
        
        # 6. Active Energy Modeling
        # DRAM fetch cost: 15 pJ/bit
        dram_energy_j = ((actual_weight_load_mb + act_spill_bytes_mb * 2.0) * 1024 * 1024 * 8) * 15e-12
        # Centralized SRAM access cost: 0.5 pJ/bit (long, high-capacitance global wires)
        sram_energy_j = ((raw_act_size_mb * 2.0) * 1024 * 1024 * 8) * 0.5e-12
        # Compute cost: 0.05 pJ/MAC (0.025 pJ/FLOP)
        compute_energy_j = layer_flops * 0.025e-12
        
        total_energy_j = dram_energy_j + sram_energy_j + compute_energy_j
        
        # Bottleneck Identification
        if act_spill_time_ms > 0.4 * latency_ms:
            bottleneck = "Activation Spill Stall (SRAM Capacity Limit)"
        elif weight_stream_time_ms > 1.2 * compute_time_ms:
            bottleneck = "HBM Weight Thrashing Memory Wall"
        elif sram_bus_delay_ms > 0.2 * latency_ms:
            bottleneck = "Centralized Global Bus Routing Contention"
        else:
            bottleneck = "Compute Pipeline Bound (Balanced Design)"
            
        return {
            'model_name': model_name,
            'model_type': model_type,
            'total_tiles': self.total_pes,
            'total_sram_mb': self.sram_capacity_mb,
            'peak_compute_tflops': self.peak_compute_tflops,
            'latency_ms': latency_ms,
            'compute_time_ms': compute_time_ms,
            'weight_stream_time_ms': weight_stream_time_ms,
            'noc_total_latency_ms': sram_bus_delay_ms, # map to bus delay for similar reporting key
            'act_spill_time_ms': act_spill_time_ms,
            'achieved_tflops': achieved_tflops,
            'pcu_utilization_pct': pcu_utilization_pct,
            'total_energy_j': total_energy_j,
            'sram_spilled': sram_spilled,
            'bottleneck': bottleneck
        }

if __name__ == '__main__':
    sim = NPUMicroSim()
    llama = {
        'model_name': 'LLaMA-3-70B',
        'type': 'dense',
        'seq_len': 8192,
        'weight_size_mb': 1856.0,
        'hidden_dim': 8192,
        'ffn_dim': 28672
    }
    res = sim.simulate_layer(llama)
    print(f"Test Run TPU-style NPU LLaMA-3-70B: {res['achieved_tflops']:.2f} TFLOPS | Bottleneck: {res['bottleneck']}")
