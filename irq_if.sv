interface irq_if(input logic clk, input logic rst_n);
  logic [31:0] m_irq; logic m_msi_valid; logic [15:0] m_msi_vector;
endinterface
