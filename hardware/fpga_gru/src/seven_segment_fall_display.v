`timescale 1ns / 1ps

module seven_segment_fall_display #(
    parameter REFRESH_COUNTER_WIDTH = 18
) (
    input  wire       clk,
    input  wire       reset,
    input  wire       show_fall,
    output reg  [6:0] seg,
    output reg  [3:0] an,
    output wire       dp
);

    reg [REFRESH_COUNTER_WIDTH-1:0] refresh_counter;
    wire [1:0] digit_select = refresh_counter[REFRESH_COUNTER_WIDTH-1:REFRESH_COUNTER_WIDTH-2];

    localparam [6:0] SEG_BLANK = 7'b1111111;
    localparam [6:0] SEG_F     = 7'b0001110;
    localparam [6:0] SEG_A     = 7'b0001000;
    localparam [6:0] SEG_L     = 7'b1000111;

    assign dp = 1'b1;

    always @(posedge clk) begin
        if (reset) begin
            refresh_counter <= {REFRESH_COUNTER_WIDTH{1'b0}};
        end else begin
            refresh_counter <= refresh_counter + 1'b1;
        end
    end

    always @* begin
        if (!show_fall) begin
            an = 4'b1111;
            seg = SEG_BLANK;
        end else begin
            case (digit_select)
                2'd0: begin
                    an = 4'b1110;
                    seg = SEG_L;
                end
                2'd1: begin
                    an = 4'b1101;
                    seg = SEG_L;
                end
                2'd2: begin
                    an = 4'b1011;
                    seg = SEG_A;
                end
                default: begin
                    an = 4'b0111;
                    seg = SEG_F;
                end
            endcase
        end
    end

endmodule
