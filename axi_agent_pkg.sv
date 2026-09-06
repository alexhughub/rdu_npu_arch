package axi_agent_pkg;
 import uvm_pkg::*; `include "uvm_macros.svh"
 class axi_txn extends uvm_sequence_item;
  `uvm_object_utils(axi_txn)
  rand bit m_write; rand bit [63:0] m_addr; rand bit [255:0] m_data; rand bit [31:0] m_strb; rand int unsigned m_bytes;
  function new(string name="axi_txn"); super.new(name); endfunction
 endclass
 class axi_sequencer extends uvm_sequencer#(axi_txn); `uvm_component_utils(axi_sequencer) function new(string name,uvm_component parent);super.new(name,parent);endfunction endclass
 class axi_driver extends uvm_driver#(axi_txn);
  `uvm_component_utils(axi_driver) virtual axi_if#(64,256,8) m_vif;
  function new(string name,uvm_component parent);super.new(name,parent);endfunction
  function void build_phase(uvm_phase phase); if(!uvm_config_db#(virtual axi_if#(64,256,8))::get(this,"","vif",m_vif)) `uvm_fatal("NOVIF","AXI vif") endfunction
  task run_phase(uvm_phase phase); axi_txn txn; forever begin seq_item_port.get_next_item(txn); `uvm_info("AXI_DRV",$sformatf("%s addr=%h bytes=%0d",txn.m_write?"WRITE":"READ",txn.m_addr,txn.m_bytes),UVM_HIGH) seq_item_port.item_done(); end endtask
 endclass
 class axi_monitor extends uvm_monitor;
  `uvm_component_utils(axi_monitor) virtual axi_if#(64,256,8) m_vif; uvm_analysis_port#(axi_txn) m_ap;
  function new(string name,uvm_component parent);super.new(name,parent);m_ap=new("ap",this);endfunction
  function void build_phase(uvm_phase phase); if(!uvm_config_db#(virtual axi_if#(64,256,8))::get(this,"","vif",m_vif)) `uvm_fatal("NOVIF","AXI vif") endfunction
  task run_phase(uvm_phase phase); forever begin @(posedge m_vif.ACLK); if(m_vif.m_AWVALID&&m_vif.m_AWREADY) begin axi_txn txn; txn=axi_txn::type_id::create("wr"); txn.m_write=1;txn.m_addr=m_vif.m_AWADDR;txn.m_bytes=(m_vif.m_AWLEN+1)*(1<<m_vif.m_AWSIZE);m_ap.write(txn);end if(m_vif.m_ARVALID&&m_vif.m_ARREADY) begin axi_txn txn; txn=axi_txn::type_id::create("rd");txn.m_write=0;txn.m_addr=m_vif.m_ARADDR;txn.m_bytes=(m_vif.m_ARLEN+1)*(1<<m_vif.m_ARSIZE);m_ap.write(txn);end end endtask
 endclass
 class axi_single_seq extends uvm_sequence#(axi_txn);
  `uvm_object_utils(axi_single_seq) axi_txn m_req_item;
  function new(string name="axi_single_seq");super.new(name);endfunction
  task body(); if(m_req_item==null)`uvm_fatal("AXI_SEQ","m_req_item null") start_item(m_req_item);finish_item(m_req_item);endtask
 endclass
 class axi_agent extends uvm_agent;
  `uvm_component_utils(axi_agent) axi_sequencer m_seqr; axi_driver m_drv; axi_monitor m_mon;
  function new(string name,uvm_component parent);super.new(name,parent);endfunction
  function void build_phase(uvm_phase phase);super.build_phase(phase);m_mon=axi_monitor::type_id::create("mon",this);if(get_is_active()==UVM_ACTIVE)begin m_seqr=axi_sequencer::type_id::create("seqr",this);m_drv=axi_driver::type_id::create("drv",this);end endfunction
  function void connect_phase(uvm_phase phase);if(get_is_active()==UVM_ACTIVE)m_drv.seq_item_port.connect(m_seqr.seq_item_export);endfunction
 endclass
endpackage
