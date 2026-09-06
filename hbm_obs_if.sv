interface hbm_obs_if #(parameter AW=64, DW=256)(input logic clk, input logic rst_n);
  logic m_valid,m_write; logic [AW-1:0] m_addr; logic [DW-1:0] m_data; logic [DW/8-1:0] m_be; logic [15:0] m_context_id;
endinterface
