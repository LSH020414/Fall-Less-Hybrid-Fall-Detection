`timescale 1ns / 1ps

module gru_result_uart_tx #(
    parameter integer CLKS_PER_BIT = 868
) (
    input  wire              clk,
    input  wire              reset,
    input  wire              result_valid,
    input  wire [15:0]       sequence_id,
    input  wire [15:0]       probability_q15,
    input  wire              fall_class,
    output wire              tx,
    output reg               busy
);

    reg [2:0] byte_index;
    reg [15:0] sequence_latched;
    reg [15:0] probability_latched;
    reg class_latched;
    reg [7:0] checksum_latched;
    reg [7:0] tx_byte;
    reg tx_start;
    wire tx_busy;
    wire tx_done;

    function automatic [7:0] selected_byte;
        input [2:0] index;
        begin
            case (index)
                3'd0: selected_byte = 8'hF1;
                3'd1: selected_byte = sequence_latched[7:0];
                3'd2: selected_byte = sequence_latched[15:8];
                3'd3: selected_byte = probability_latched[7:0];
                3'd4: selected_byte = probability_latched[15:8];
                3'd5: selected_byte = {7'd0, class_latched};
                default: selected_byte = checksum_latched;
            endcase
        end
    endfunction

    uart_tx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_uart_tx (
        .clk(clk),
        .reset(reset),
        .tx_byte(tx_byte),
        .tx_start(tx_start),
        .tx(tx),
        .tx_busy(tx_busy),
        .tx_done(tx_done)
    );

    always @(posedge clk) begin
        if (reset) begin
            byte_index <= 3'd0;
            sequence_latched <= 16'd0;
            probability_latched <= 16'd0;
            class_latched <= 1'b0;
            checksum_latched <= 8'd0;
            tx_byte <= 8'd0;
            tx_start <= 1'b0;
            busy <= 1'b0;
        end else begin
            tx_start <= 1'b0;

            if (!busy && result_valid) begin
                sequence_latched <= sequence_id;
                probability_latched <= probability_q15;
                class_latched <= fall_class;
                checksum_latched <=
                    8'hF1 ^
                    sequence_id[7:0] ^
                    sequence_id[15:8] ^
                    probability_q15[7:0] ^
                    probability_q15[15:8] ^
                    {7'd0, fall_class};
                byte_index <= 3'd0;
                busy <= 1'b1;
            end else if (busy && !tx_busy && !tx_start && !tx_done) begin
                tx_byte <= selected_byte(byte_index);
                tx_start <= 1'b1;
            end

            if (busy && tx_done) begin
                if (byte_index == 3'd6) begin
                    busy <= 1'b0;
                end else begin
                    byte_index <= byte_index + 1'b1;
                end
            end
        end
    end

endmodule
