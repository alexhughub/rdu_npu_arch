interface axi_if #(parameter AW=64, DW=256, IW=8)(input logic ACLK, input logic ARESETn);
  logic [IW-1:0] m_AWID; logic [AW-1:0] m_AWADDR; logic [7:0] m_AWLEN; logic [2:0] m_AWSIZE; logic [1:0] m_AWBURST; logic m_AWVALID,m_AWREADY;
  logic [DW-1:0] m_WDATA; logic [DW/8-1:0] m_WSTRB; logic m_WLAST,m_WVALID,m_WREADY;
  logic [IW-1:0] m_BID; logic [1:0] m_BRESP; logic m_BVALID,m_BREADY;
  logic [IW-1:0] m_ARID; logic [AW-1:0] m_ARADDR; logic [7:0] m_ARLEN; logic [2:0] m_ARSIZE; logic [1:0] m_ARBURST; logic m_ARVALID,m_ARREADY;
  logic [IW-1:0] m_RID; logic [DW-1:0] m_RDATA; logic [1:0] m_RRESP; logic m_RLAST,m_RVALID,m_RREADY;
endinterface
