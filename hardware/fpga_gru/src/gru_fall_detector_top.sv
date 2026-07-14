`timescale 1ns / 1ps

module gru_fall_detector_top #(
    parameter integer CLKS_PER_BIT = 868,
    parameter integer WINDOWS_PER_CHUNK = 17,
    parameter integer HIGH_THRESHOLD_Q15 = 22938,
    parameter integer MIN_HIGH_WINDOWS = 2,
    parameter integer AVG_SUM_THRESHOLD_Q15 = 278528,
    parameter integer MOTION_THRESHOLD_Q14 = 1147
) (
    input  wire       clk,
    input  wire       reset,
    input  wire       uart_rx_i,
    output wire       uart_tx_o,
    output wire       led,
    output wire [6:0] seg,
    output wire [3:0] an,
    output wire       dp
);

    wire [7:0] rx_byte;
    wire rx_valid;
    wire core_input_we;
    wire [9:0] core_input_addr;
    wire signed [17:0] core_input_data;
    wire core_start;
    wire [15:0] parser_sequence_id;
    wire checksum_error;
    wire packet_valid;
    wire core_busy;
    wire core_done;
    wire [15:0] probability_q15;
    wire signed [31:0] logit0_q12;
    wire signed [31:0] logit1_q12;
    wire result_tx_busy;
    wire [39:0] packet_motion_sum;
    wire [15:0] packet_motion_pairs;
    wire first_window_in_chunk;
    wire chunk_result_class;
    wire chunk_complete;
    wire fall_alert;
    wire [4:0] window_count_debug;
    wire [5:0] high_count_debug;

    reg [15:0] active_sequence_id;

    uart_rx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_uart_rx (
        .clk(clk),
        .reset(reset),
        .rx(uart_rx_i),
        .rx_byte(rx_byte),
        .rx_valid(rx_valid)
    );

    gru_sequence_uart_parser u_gru_sequence_uart_parser (
        .clk(clk),
        .reset(reset),
        .rx_byte(rx_byte),
        .rx_valid(rx_valid),
        .core_busy(core_busy),
        .input_we(core_input_we),
        .input_addr(core_input_addr),
        .input_data(core_input_data),
        .start(core_start),
        .sequence_id(parser_sequence_id),
        .checksum_error(checksum_error),
        .packet_valid(packet_valid)
    );

    gru_fixed_core u_gru_fixed_core (
        .clk(clk),
        .reset(reset),
        .input_we(core_input_we),
        .input_addr(core_input_addr),
        .input_data(core_input_data),
        .start(core_start),
        .busy(core_busy),
        .done(core_done),
        .fall_probability_q15(probability_q15),
        .logit0_q12(logit0_q12),
        .logit1_q12(logit1_q12)
    );

    gru_motion_tracker u_gru_motion_tracker (
        .clk(clk),
        .reset(reset),
        .input_we(core_input_we),
        .input_addr(core_input_addr),
        .input_data(core_input_data),
        .first_window_in_chunk(first_window_in_chunk),
        .packet_motion_sum(packet_motion_sum),
        .packet_motion_pairs(packet_motion_pairs)
    );

    gru_chunk_decision #(
        .WINDOWS_PER_CHUNK(WINDOWS_PER_CHUNK),
        .HIGH_THRESHOLD_Q15(HIGH_THRESHOLD_Q15),
        .MIN_HIGH_WINDOWS(MIN_HIGH_WINDOWS),
        .AVG_SUM_THRESHOLD_Q15(AVG_SUM_THRESHOLD_Q15),
        .MOTION_THRESHOLD_Q14(MOTION_THRESHOLD_Q14)
    ) u_gru_chunk_decision (
        .clk(clk),
        .reset(reset),
        .window_done(core_done),
        .probability_q15(probability_q15),
        .packet_motion_sum(packet_motion_sum),
        .packet_motion_pairs(packet_motion_pairs),
        .first_window_in_chunk(first_window_in_chunk),
        .result_class(chunk_result_class),
        .chunk_complete(chunk_complete),
        .fall_alert(fall_alert),
        .window_count_debug(window_count_debug),
        .high_count_debug(high_count_debug)
    );

    gru_result_uart_tx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_gru_result_uart_tx (
        .clk(clk),
        .reset(reset),
        .result_valid(core_done),
        .sequence_id(active_sequence_id),
        .probability_q15(probability_q15),
        .fall_class(chunk_result_class),
        .tx(uart_tx_o),
        .busy(result_tx_busy)
    );

    seven_segment_fall_display u_seven_segment_fall_display (
        .clk(clk),
        .reset(reset),
        .show_fall(fall_alert),
        .seg(seg),
        .an(an),
        .dp(dp)
    );

    assign led = fall_alert;

    always @(posedge clk) begin
        if (reset) begin
            active_sequence_id <= 16'd0;
        end else begin
            if (core_start)
                active_sequence_id <= parser_sequence_id;
        end
    end

endmodule
