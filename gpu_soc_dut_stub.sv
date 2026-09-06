module gpu_soc_dut_stub(input logic clk, input logic por_n, input logic gpu_reset_n, input logic cpu_reset_n);
  logic [7:0] m_host_awid; logic [63:0] m_host_awaddr; logic [7:0] m_host_awlen; logic [2:0] m_host_awsize; logic [1:0] m_host_awburst; logic m_host_awvalid,m_host_awready;
  logic [255:0] m_host_wdata; logic [31:0] m_host_wstrb; logic m_host_wlast,m_host_wvalid,m_host_wready; logic [7:0] m_host_bid; logic [1:0] m_host_bresp; logic m_host_bvalid,m_host_bready;
  logic [7:0] m_host_arid; logic [63:0] m_host_araddr; logic [7:0] m_host_arlen; logic [2:0] m_host_arsize; logic [1:0] m_host_arburst; logic m_host_arvalid,m_host_arready;
  logic [7:0] m_host_rid; logic [255:0] m_host_rdata; logic [1:0] m_host_rresp; logic m_host_rlast,m_host_rvalid,m_host_rready;
  logic [31:0] m_apb_paddr,m_apb_pwdata,m_apb_prdata; logic m_apb_psel,m_apb_penable,m_apb_pwrite,m_apb_pready,m_apb_pslverr;
  logic [31:0] m_irq; logic m_msi_valid; logic [15:0] m_msi_vector;
  logic m_hbm_valid,m_hbm_write; logic [63:0] m_hbm_addr; logic [255:0] m_hbm_data; logic [31:0] m_hbm_be; logic [15:0] m_hbm_context_id;
endmodule
