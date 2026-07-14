`timescale 1ns / 1ps

module gru_chunk_decision #(
    parameter integer WINDOWS_PER_CHUNK = 17,
    parameter integer HIGH_THRESHOLD_Q15 = 22938,
    parameter integer MIN_HIGH_WINDOWS = 2,
    parameter integer AVG_SUM_THRESHOLD_Q15 = 278528,
    parameter integer MOTION_THRESHOLD_Q14 = 1147
) (
    input  wire          clk,
    input  wire          reset,
    input  wire          window_done,
    input  wire [15:0]   probability_q15,
    input  wire [39:0]   packet_motion_sum,
    input  wire [15:0]   packet_motion_pairs,
    output wire          first_window_in_chunk,
    output wire          result_class,
    output reg           chunk_complete,
    output reg           fall_alert,
    output reg [4:0]     window_count_debug,
    output reg [5:0]     high_count_debug
);

    reg [20:0] probability_sum;
    reg [39:0] motion_sum;
    reg [15:0] motion_pairs;

    wire current_high = probability_q15 >= HIGH_THRESHOLD_Q15;
    wire [5:0] final_high_count =
        high_count_debug + (current_high ? 1'b1 : 1'b0);
    wire [20:0] final_probability_sum =
        probability_sum + probability_q15;
    wire [39:0] final_motion_sum =
        motion_sum + packet_motion_sum;
    wire [16:0] final_motion_pairs =
        motion_pairs + packet_motion_pairs;
    wire final_window =
        window_count_debug == WINDOWS_PER_CHUNK - 1;

    wire probability_condition =
        (final_high_count >= MIN_HIGH_WINDOWS) &&
        (final_probability_sum >= AVG_SUM_THRESHOLD_Q15);

    wire [39:0] motion_threshold_total =
        final_motion_pairs * MOTION_THRESHOLD_Q14;

    wire motion_condition =
        (final_motion_pairs != 0) &&
        (final_motion_sum >= motion_threshold_total);

    wire final_fall =
        final_window && probability_condition && motion_condition;

    assign first_window_in_chunk = window_count_debug == 0;
    assign result_class = window_done && final_fall;

    always @(posedge clk) begin
        if (reset) begin
            probability_sum <= 0;
            motion_sum <= 0;
            motion_pairs <= 0;
            window_count_debug <= 0;
            high_count_debug <= 0;
            chunk_complete <= 0;
            fall_alert <= 0;
        end else begin
            chunk_complete <= 0;

            if (window_done) begin
                if (final_window) begin
                    chunk_complete <= 1;
                    if (final_fall)
                        fall_alert <= 1;

                    probability_sum <= 0;
                    motion_sum <= 0;
                    motion_pairs <= 0;
                    window_count_debug <= 0;
                    high_count_debug <= 0;
                end else begin
                    probability_sum <= final_probability_sum;
                    motion_sum <= final_motion_sum;
                    motion_pairs <= final_motion_pairs[15:0];
                    window_count_debug <= window_count_debug + 1'b1;
                    high_count_debug <= final_high_count;
                end
            end
        end
    end

endmodule
