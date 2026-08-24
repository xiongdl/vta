module ahb32_to_apb32 #(
  parameter int ADDR_WIDTH = 32
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic [ADDR_WIDTH-1:0] haddr,
  input  logic [1:0]            htrans,
  input  logic                  hwrite,
  input  logic [2:0]            hsize,
  input  logic [31:0]           hwdata,
  output logic [31:0]           hrdata,
  output logic                  hready,
  output logic                  hresp,
  output logic [ADDR_WIDTH-1:0] paddr,
  output logic                  psel,
  output logic                  penable,
  output logic                  pwrite,
  output logic [31:0]           pwdata,
  output logic [3:0]            pstrb,
  output logic [2:0]            pprot,
  input  logic [31:0]           prdata,
  input  logic                  pready,
  input  logic                  pslverr
);
  typedef enum logic [2:0] {IDLE, WRITE_DATA, SETUP, ACCESS, ERROR_1, ERROR_2} state_t;
  state_t state;

  function automatic logic [3:0] byte_strobe(input logic [2:0] size, input logic [1:0] addr);
    case (size)
      3'd0: byte_strobe = 4'b0001 << addr;
      3'd1: byte_strobe = addr[1] ? 4'b1100 : 4'b0011;
      3'd2: byte_strobe = 4'b1111;
      default: byte_strobe = 4'b0000;
    endcase
  endfunction

  always_comb begin
    psel    = (state == SETUP) || (state == ACCESS);
    penable = (state == ACCESS);
    hready  = (state == IDLE) || (state == ERROR_2);
    hresp   = (state == ERROR_1) || (state == ERROR_2);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state  <= IDLE;
      paddr  <= '0;
      pwrite <= 1'b0;
      pwdata <= '0;
      pstrb  <= '0;
      pprot  <= '0;
      hrdata <= '0;
    end else begin
      case (state)
        IDLE: if (htrans[1]) begin
          paddr  <= haddr;
          pwrite <= hwrite;
          pstrb  <= hwrite ? byte_strobe(hsize, haddr[1:0]) : 4'b0000;
          pprot  <= 3'b000;
          if (hsize > 3'd2)
            state <= ERROR_1;
          else
            state <= hwrite ? WRITE_DATA : SETUP;
        end
        // AHB write data belongs to the data phase following the accepted
        // address phase, so it must not be sampled together with HADDR.
        WRITE_DATA: begin
          pwdata <= hwdata;
          state  <= SETUP;
        end
        SETUP: state <= ACCESS;
        ACCESS: if (pready) begin
          hrdata <= prdata;
          state  <= pslverr ? ERROR_1 : IDLE;
        end
        ERROR_1: state <= ERROR_2;
        ERROR_2: state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end
endmodule
