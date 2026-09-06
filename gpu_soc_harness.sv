module gpu_soc_harness(
  input logic clk,por_n,gpu_reset_n,cpu_reset_n,
  input logic [7:0] host_awid, input logic [63:0] host_awaddr, input logic [7:0] host_awlen, input logic [2:0] host_awsize, input logic [1:0] host_awburst, input logic host_awvalid,host_awready,
  input logic [255:0] host_wdata, input logic [31:0] host_wstrb, input logic host_wlast,host_wvalid,host_wready, input logic [7:0] host_bid, input logic [1:0] host_bresp, input logic host_bvalid,host_bready,
  input logic [7:0] host_arid, input logic [63:0] host_araddr, input logic [7:0] host_arlen, input logic [2:0] host_arsize, input logic [1:0] host_arburst, input logic host_arvalid,host_arready,
  input logic [7:0] host_rid, input logic [255:0] host_rdata, input logic [1:0] host_rresp, input logic host_rlast,host_rvalid,host_rready,
  input logic [31:0] apb_paddr, input logic apb_psel,apb_penable,apb_pwrite, input logic [31:0] apb_pwdata,apb_prdata, input logic apb_pready,apb_pslverr,
  input logic [31:0] irq, input logic msi_valid, input logic [15:0] msi_vector,
  input logic hbm_valid,hbm_write, input logic [63:0] hbm_addr, input logic [255:0] hbm_data, input logic [31:0] hbm_be, input logic [15:0] hbm_context_id
);
  soc_clk_reset_if m_rst_vif(clk);
  axi_if #(.AW(64),.DW(256),.IW(8)) m_host_axi_vif(clk,gpu_reset_n);
  apb_if #(.AW(32),.DW(32)) m_mgmt_apb_vif(clk,gpu_reset_n);
  irq_if m_irq_vif(clk,gpu_reset_n);
  hbm_obs_if #(.AW(64),.DW(256)) m_hbm_vif(clk,gpu_reset_n);
  assign m_rst_vif.m_por_n=por_n; assign m_rst_vif.m_gpu_reset_n=gpu_reset_n; assign m_rst_vif.m_cpu_reset_n=cpu_reset_n;
  assign m_host_axi_vif.m_AWID=host_awid; assign m_host_axi_vif.m_AWADDR=host_awaddr; assign m_host_axi_vif.m_AWLEN=host_awlen; assign m_host_axi_vif.m_AWSIZE=host_awsize; assign m_host_axi_vif.m_AWBURST=host_awburst; assign m_host_axi_vif.m_AWVALID=host_awvalid; assign m_host_axi_vif.m_AWREADY=host_awready;
  assign m_host_axi_vif.m_WDATA=host_wdata; assign m_host_axi_vif.m_WSTRB=host_wstrb; assign m_host_axi_vif.m_WLAST=host_wlast; assign m_host_axi_vif.m_WVALID=host_wvalid; assign m_host_axi_vif.m_WREADY=host_wready;
  assign m_host_axi_vif.m_BID=host_bid; assign m_host_axi_vif.m_BRESP=host_bresp; assign m_host_axi_vif.m_BVALID=host_bvalid; assign m_host_axi_vif.m_BREADY=host_bready;
  assign m_host_axi_vif.m_ARID=host_arid; assign m_host_axi_vif.m_ARADDR=host_araddr; assign m_host_axi_vif.m_ARLEN=host_arlen; assign m_host_axi_vif.m_ARSIZE=host_arsize; assign m_host_axi_vif.m_ARBURST=host_arburst; assign m_host_axi_vif.m_ARVALID=host_arvalid; assign m_host_axi_vif.m_ARREADY=host_arready;
  assign m_host_axi_vif.m_RID=host_rid; assign m_host_axi_vif.m_RDATA=host_rdata; assign m_host_axi_vif.m_RRESP=host_rresp; assign m_host_axi_vif.m_RLAST=host_rlast; assign m_host_axi_vif.m_RVALID=host_rvalid; assign m_host_axi_vif.m_RREADY=host_rready;
  assign m_mgmt_apb_vif.m_PADDR=apb_paddr; assign m_mgmt_apb_vif.m_PSEL=apb_psel; assign m_mgmt_apb_vif.m_PENABLE=apb_penable; assign m_mgmt_apb_vif.m_PWRITE=apb_pwrite; assign m_mgmt_apb_vif.m_PWDATA=apb_pwdata; assign m_mgmt_apb_vif.m_PRDATA=apb_prdata; assign m_mgmt_apb_vif.m_PREADY=apb_pready; assign m_mgmt_apb_vif.m_PSLVERR=apb_pslverr;
  assign m_irq_vif.m_irq=irq; assign m_irq_vif.m_msi_valid=msi_valid; assign m_irq_vif.m_msi_vector=msi_vector;
  assign m_hbm_vif.m_valid=hbm_valid; assign m_hbm_vif.m_write=hbm_write; assign m_hbm_vif.m_addr=hbm_addr; assign m_hbm_vif.m_data=hbm_data; assign m_hbm_vif.m_be=hbm_be; assign m_hbm_vif.m_context_id=hbm_context_id;
endmodule
