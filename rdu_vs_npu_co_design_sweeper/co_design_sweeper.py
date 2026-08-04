#!/usr/bin/env python3
"""
High-Level Python Extreme Regimes Co-Design Sweeper.
Compares SambaNova Spatial RDU vs. TPU-style Centralized Systolic NPU 
under Training Large-Batch (B=128, S=512) and Real-Time Serving (B=1, S=32k).
Outputs results to python_sweep_results.csv.
"""

import os
import json
import pandas as pd
from datetime import datetime

class RDUSimulator:
    def __init__(self, grid_size=32, sram_per_pmu_kb=128, hbm_bw=2400.0, noc_bw=256.0):
        self.grid_size = grid_size
        self.total_tiles = grid_size * grid_size
        self.peak_compute_tflops = (self.total_tiles * 1024.0 * 1e9) / 1e12 # 1GHz base clock
        self.total_sram_mb = (self.total_tiles * sram_per_pmu_kb) / 1024.0
        self.hbm_bandwidth = hbm_bw
        self.noc_bandwidth = noc_bw
        self.silicon_cost = 37.59 if grid_size == 32 else 25.36

    def simulate(self, wl):
        batch = wl['batch_size']
        seq_len = wl['seq_len']
        hidden_dim = wl['hidden_dim']
        ffn_dim = wl['ffn_dim']
        weight_size_mb = wl['weight_size_mb']
        
        # Flops estimation
        if wl['model_type'] == 'dense':
            layer_flops = 2.0 * batch * (
                3.0 * seq_len * hidden_dim * hidden_dim +
                2.0 * seq_len * seq_len * hidden_dim +
                seq_len * hidden_dim * hidden_dim +
                3.0 * seq_len * hidden_dim * ffn_dim
            )
        else: # moe
            layer_flops = 2.0 * batch * (
                3.0 * seq_len * hidden_dim * hidden_dim +
                seq_len * hidden_dim * ffn_dim * 2.0 * (wl['num_shared_experts'] + wl['routed_experts_per_token'])
            )
        layer_gflops = layer_flops / 1e9
        
        # High level compute time (assuming perfect 92% utilization, no bank conflict stalls)
        compute_time_ms = layer_gflops / (self.peak_compute_tflops * 0.92)
        
        # Activation Sizing and SRAM
        raw_act_size_mb = batch * (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0)
        # INT4 stream compression on RDU (4x effective capacity)
        compressed_act_size_mb = raw_act_size_mb / 4.0
        
        spilled = compressed_act_size_mb > (self.total_sram_mb * 0.70)
        spill_bytes_mb = 0.0
        spill_time_ms = 0.0
        if spilled:
            spill_bytes_mb = compressed_act_size_mb - (self.total_sram_mb * 0.70)
            spill_time_ms = (spill_bytes_mb * 2.0) / (self.hbm_bandwidth * 1000.0 / 1024.0)
            
        # Weight Stream time
        weight_load_ms = weight_size_mb / (self.hbm_bandwidth * 1000.0 / 1024.0)
        
        # High level flat NoC routing (ideal Manhattan delay, no congestion backpressure)
        hops = self.grid_size * (2.0 / 3.0)
        noc_routing_ms = (hops * 2e-6) + (compressed_act_size_mb / (self.noc_bandwidth * 1000.0 / 1024.0))
        
        # High level overlap factor
        prefetch_overlap = 0.94
        latency_ms = max(compute_time_ms, weight_load_ms) + weight_load_ms * (1.0 - prefetch_overlap) + noc_routing_ms + spill_time_ms
        
        achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0)
        utilization = (achieved_tflops / self.peak_compute_tflops) * 100.0
        tflops_per_dollar = achieved_tflops / self.silicon_cost
        
        # Energy
        dram_energy = ((weight_size_mb + spill_bytes_mb * 2.0) * 1024 * 1024 * 8) * 15e-12
        sram_energy = ((compressed_act_size_mb * 2.0) * 1024 * 1024 * 8) * 0.1e-12
        compute_energy = layer_flops * 0.025e-12
        total_energy = dram_energy + sram_energy + compute_energy
        
        return {
            'model_name': wl['model_name'],
            'arch_type': 'RDU',
            'latency_ms': latency_ms,
            'achieved_tflops': achieved_tflops,
            'pcu_utilization_pct': utilization,
            'total_energy_j': total_energy,
            'sram_spilled': 'Yes' if spilled else 'No',
            'tflops_per_dollar': tflops_per_dollar,
            'bottleneck': 'Compute Pipeline Bound' if not spilled else 'Activation Spill Stall'
        }


class NPUSimulator:
    def __init__(self, grid_size=712, sram_capacity_mb=256.0, hbm_bw=2400.0, bus_bw=4800.0):
        self.grid_size = grid_size
        self.total_pes = grid_size * grid_size
        self.peak_compute_tflops = (self.total_pes * 2.0 * 1e9) / 1e12 # 1GHz clock
        self.sram_capacity_mb = sram_capacity_mb
        self.hbm_bandwidth = hbm_bw
        self.bus_bandwidth = bus_bw
        self.silicon_cost = 22.01 if grid_size == 712 else 22.52

    def simulate(self, wl):
        batch = wl['batch_size']
        seq_len = wl['seq_len']
        hidden_dim = wl['hidden_dim']
        ffn_dim = wl['ffn_dim']
        weight_size_mb = wl['weight_size_mb']
        
        # Flops estimation
        if wl['model_type'] == 'dense':
            layer_flops = 2.0 * batch * (
                3.0 * seq_len * hidden_dim * hidden_dim +
                2.0 * seq_len * seq_len * hidden_dim +
                seq_len * hidden_dim * hidden_dim +
                3.0 * seq_len * hidden_dim * ffn_dim
            )
        else: # moe
            layer_flops = 2.0 * batch * (
                3.0 * seq_len * hidden_dim * hidden_dim +
                seq_len * hidden_dim * ffn_dim * 2.0 * (wl['num_shared_experts'] + wl['routed_experts_per_token'])
            )
        layer_gflops = layer_flops / 1e9
        
        # High level compute time (perfect 96% utilization, no setup phase, no bus contention)
        compute_time_ms = layer_gflops / (self.peak_compute_tflops * 0.96)
        
        # Activation Sizing and Spilling (No compression)
        raw_act_size_mb = batch * (seq_len * hidden_dim * 2.0 * 4.0) / (1024.0 * 1024.0)
        spilled = raw_act_size_mb > (self.sram_capacity_mb * 0.70)
        spill_bytes_mb = 0.0
        spill_time_ms = 0.0
        if spilled:
            spill_bytes_mb = raw_act_size_mb - (self.sram_capacity_mb * 0.70)
            spill_time_ms = (spill_bytes_mb * 2.0) / (self.hbm_bandwidth * 1000.0 / 1024.0)
            
        # Weight Stream (with MoE weight thrashing)
        actual_weight_load_mb = weight_size_mb
        if wl['model_type'] == 'moe':
            actual_weight_load_mb = weight_size_mb * (wl['routed_experts_per_token'] / 2.0)
            
        weight_load_ms = actual_weight_load_mb / (self.hbm_bandwidth * 1000.0 / 1024.0)
        
        # High level flat global bus delay (no Row contention)
        sram_bus_delay_ms = (raw_act_size_mb + actual_weight_load_mb) / (self.bus_bandwidth * 1000.0 / 1024.0)
        
        # Shared ports limits overlap
        prefetch_overlap = 0.10
        latency_ms = compute_time_ms + weight_load_ms * (1.0 - prefetch_overlap) + sram_bus_delay_ms + spill_time_ms
        
        achieved_tflops = (layer_gflops / 1000.0) / (latency_ms / 1000.0)
        utilization = (achieved_tflops / self.peak_compute_tflops) * 100.0
        tflops_per_dollar = achieved_tflops / self.silicon_cost
        
        # Energy (High central bus wire cost)
        dram_energy = ((actual_weight_load_mb + spill_bytes_mb * 2.0) * 1024 * 1024 * 8) * 15e-12
        sram_energy = ((raw_act_size_mb * 2.0) * 1024 * 1024 * 8) * 0.5e-12
        compute_energy = layer_flops * 0.025e-12
        total_energy = dram_energy + sram_energy + compute_energy
        
        return {
            'model_name': wl['model_name'],
            'arch_type': 'NPU',
            'latency_ms': latency_ms,
            'achieved_tflops': achieved_tflops,
            'pcu_utilization_pct': utilization,
            'total_energy_j': total_energy,
            'sram_spilled': 'Yes' if spilled else 'No',
            'tflops_per_dollar': tflops_per_dollar,
            'bottleneck': 'Compute Pipeline Bound' if not spilled else 'Activation Spill Stall'
        }


def main():
    workloads = [
        { "model_name": "LLaMA-3-70B (Training Large-Batch)", "model_type": "dense", "batch_size": 128.0, "seq_len": 512.0, "weight_size_mb": 1856.0, "hidden_dim": 8192.0, "ffn_dim": 28672.0, "num_shared_experts": 0.0, "num_routed_experts": 0.0, "routed_experts_per_token": 0.0 },
        { "model_name": "DeepSeek-V3 (Training Large-Batch)", "model_type": "moe", "batch_size": 128.0, "seq_len": 512.0, "weight_size_mb": 1150.0, "hidden_dim": 7168.0, "ffn_dim": 2048.0, "num_shared_experts": 1.0, "num_routed_experts": 256.0, "routed_experts_per_token": 8.0 },
        { "model_name": "LLaMA-3-70B (Serving Extreme Context)", "model_type": "dense", "batch_size": 1.0, "seq_len": 32768.0, "weight_size_mb": 1856.0, "hidden_dim": 8192.0, "ffn_dim": 28672.0, "num_shared_experts": 0.0, "num_routed_experts": 0.0, "routed_experts_per_token": 0.0 },
        { "model_name": "DeepSeek-V3 (Serving Extreme Context)", "model_type": "moe", "batch_size": 1.0, "seq_len": 32768.0, "weight_size_mb": 1150.0, "hidden_dim": 7168.0, "ffn_dim": 2048.0, "num_shared_experts": 1.0, "num_routed_experts": 256.0, "routed_experts_per_token": 8.0 }
    ]

    rdu = RDUSimulator(32, 128.0, 2400.0, 256.0)
    npu = NPUSimulator(712, 256.0, 2400.0, 4800.0)
    
    results = []
    print("[+] Running High-Level Python Co-Design comparison sweeps...")
    
    for wl in workloads:
        results.append(rdu.simulate(wl))
        results.append(npu.simulate(wl))
        
    df = pd.DataFrame(results)
    df.to_csv("co_design_python_sweep_results.csv", index=False)
    print("[+] High-Level Python sweeps exported to: co_design_python_sweep_results.csv")

if __name__ == '__main__':
    main()
