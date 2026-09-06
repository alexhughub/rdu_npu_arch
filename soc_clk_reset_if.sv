interface soc_clk_reset_if(input logic clk);
  logic m_por_n,m_gpu_reset_n,m_cpu_reset_n;
  clocking cb @(posedge clk); output m_por_n,m_gpu_reset_n,m_cpu_reset_n; endclocking
endinterface
