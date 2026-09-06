`timescale 1ns/1ps
module tb_top;import uvm_pkg::*;import gpu_soc_test_pkg::*;
 logic m_clk=0,m_por_n=0,m_gpu_reset_n=0,m_cpu_reset_n=0;always #0.25 m_clk=~m_clk;
 gpu_soc_dut_stub m_dut(.clk(m_clk),.por_n(m_por_n),.gpu_reset_n(m_gpu_reset_n),.cpu_reset_n(m_cpu_reset_n));
 bind gpu_soc_dut_stub gpu_soc_harness u_gpu_soc_harness(.clk(clk),.por_n(por_n),.gpu_reset_n(gpu_reset_n),.cpu_reset_n(cpu_reset_n),
 .host_awid(m_host_awid),.host_awaddr(m_host_awaddr),.host_awlen(m_host_awlen),.host_awsize(m_host_awsize),.host_awburst(m_host_awburst),.host_awvalid(m_host_awvalid),.host_awready(m_host_awready),.host_wdata(m_host_wdata),.host_wstrb(m_host_wstrb),.host_wlast(m_host_wlast),.host_wvalid(m_host_wvalid),.host_wready(m_host_wready),.host_bid(m_host_bid),.host_bresp(m_host_bresp),.host_bvalid(m_host_bvalid),.host_bready(m_host_bready),.host_arid(m_host_arid),.host_araddr(m_host_araddr),.host_arlen(m_host_arlen),.host_arsize(m_host_arsize),.host_arburst(m_host_arburst),.host_arvalid(m_host_arvalid),.host_arready(m_host_arready),.host_rid(m_host_rid),.host_rdata(m_host_rdata),.host_rresp(m_host_rresp),.host_rlast(m_host_rlast),.host_rvalid(m_host_rvalid),.host_rready(m_host_rready),.apb_paddr(m_apb_paddr),.apb_psel(m_apb_psel),.apb_penable(m_apb_penable),.apb_pwrite(m_apb_pwrite),.apb_pwdata(m_apb_pwdata),.apb_prdata(m_apb_prdata),.apb_pready(m_apb_pready),.apb_pslverr(m_apb_pslverr),.irq(m_irq),.msi_valid(m_msi_valid),.msi_vector(m_msi_vector),.hbm_valid(m_hbm_valid),.hbm_write(m_hbm_write),.hbm_addr(m_hbm_addr),.hbm_data(m_hbm_data),.hbm_be(m_hbm_be),.hbm_context_id(m_hbm_context_id));
 initial begin
  uvm_config_db#(virtual soc_clk_reset_if)::set(null,"uvm_test_top.env","rst_vif",m_dut.u_gpu_soc_harness.m_rst_vif);
  uvm_config_db#(virtual axi_if#(64,256,8))::set(null,"uvm_test_top.env.host_axi_agent*","vif",m_dut.u_gpu_soc_harness.m_host_axi_vif);
  uvm_config_db#(virtual apb_if#(32,32))::set(null,"uvm_test_top.env.mgmt_apb_agent*","vif",m_dut.u_gpu_soc_harness.m_mgmt_apb_vif);
  uvm_config_db#(virtual irq_if)::set(null,"uvm_test_top.env.irq_agent*","vif",m_dut.u_gpu_soc_harness.m_irq_vif);
  uvm_config_db#(virtual hbm_obs_if#(64,256))::set(null,"uvm_test_top.env.hbm_agent*","vif",m_dut.u_gpu_soc_harness.m_hbm_vif);
  run_test();
 end
endmodule
