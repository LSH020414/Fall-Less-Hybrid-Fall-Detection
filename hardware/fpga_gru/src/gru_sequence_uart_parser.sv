`timescale 1ns / 1ps

module gru_sequence_uart_parser #(
    parameter integer FEATURE_COUNT = 1020
) (
    input  wire                clk,
    input  wire                reset,
    input  wire [7:0]          rx_byte,
    input  wire                rx_valid,
    input  wire                core_busy,
    output reg                 input_we,
    output reg [9:0]           input_addr,
    output reg signed [17:0]   input_data,
    output reg                 start,
    output reg [15:0]          sequence_id,
    output reg                 checksum_error,
    output reg                 packet_valid
);

    localparam [3:0]
        S_SOF0     = 4'd0,
        S_SOF1     = 4'd1,
        S_SEQ_LO   = 4'd2,
        S_SEQ_HI   = 4'd3,
        S_DATA_LO  = 4'd4,
        S_DATA_MID = 4'd5,
        S_DATA_HI  = 4'd6,
        S_CHECKSUM = 4'd7;

    reg [3:0] state;
    reg [7:0] checksum;
    reg [7:0] data_lo;
    reg [7:0] data_mid;
    reg [9:0] feature_index;

    always @(posedge clk) begin
        if (reset) begin
            state <= S_SOF0;
            checksum <= 8'd0;
            data_lo <= 8'd0;
            data_mid <= 8'd0;
            input_we <= 1'b0;
            input_addr <= 10'd0;
            input_data <= 18'sd0;
            feature_index <= 10'd0;
            start <= 1'b0;
            sequence_id <= 16'd0;
            checksum_error <= 1'b0;
            packet_valid <= 1'b0;
        end else begin
            input_we <= 1'b0;
            start <= 1'b0;
            packet_valid <= 1'b0;

            if (rx_valid) begin
                case (state)
                    S_SOF0: begin
                        if (rx_byte == 8'hA5)
                            state <= S_SOF1;
                    end

                    S_SOF1: begin
                        if (rx_byte == 8'h5A) begin
                            checksum <= 8'd0;
                            feature_index <= 10'd0;
                            state <= S_SEQ_LO;
                        end else if (rx_byte != 8'hA5) begin
                            state <= S_SOF0;
                        end
                    end

                    S_SEQ_LO: begin
                        sequence_id[7:0] <= rx_byte;
                        checksum <= rx_byte;
                        state <= S_SEQ_HI;
                    end

                    S_SEQ_HI: begin
                        sequence_id[15:8] <= rx_byte;
                        checksum <= checksum ^ rx_byte;
                        state <= S_DATA_LO;
                    end

                    S_DATA_LO: begin
                        data_lo <= rx_byte;
                        checksum <= checksum ^ rx_byte;
                        state <= S_DATA_MID;
                    end

                    S_DATA_MID: begin
                        data_mid <= rx_byte;
                        checksum <= checksum ^ rx_byte;
                        state <= S_DATA_HI;
                    end

                    S_DATA_HI: begin
                        input_data <= {rx_byte[1:0], data_mid, data_lo};
                        input_addr <= feature_index;
                        input_we <= 1'b1;
                        checksum <= checksum ^ rx_byte;
                        if (feature_index == FEATURE_COUNT - 1) begin
                            state <= S_CHECKSUM;
                        end else begin
                            feature_index <= feature_index + 1'b1;
                            state <= S_DATA_LO;
                        end
                    end

                    S_CHECKSUM: begin
                        packet_valid <= (rx_byte == checksum);
                        checksum_error <= (rx_byte != checksum);
                        if ((rx_byte == checksum) && !core_busy)
                            start <= 1'b1;
                        state <= S_SOF0;
                    end

                    default: state <= S_SOF0;
                endcase
            end
        end
    end

endmodule
