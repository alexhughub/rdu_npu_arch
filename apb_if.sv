interface apb_if #(parameter AW=32, DW=32)(input logic PCLK, input logic PRESETn);
  logic [AW-1:0] m_PADDR; logic m_PSEL,m_PENABLE,m_PWRITE; logic [DW-1:0] m_PWDATA,m_PRDATA; logic m_PREADY,m_PSLVERR;
endinterface
