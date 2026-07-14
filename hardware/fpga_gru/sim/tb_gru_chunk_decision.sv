`timescale 1ns / 1ps

module tb_gru_chunk_decision;

    reg clk;
    reg reset;
    reg window_done;
    reg [15:0] probability_q15;
    reg [39:0] packet_motion_sum;
    reg [15:0] packet_motion_pairs;

    wire first_window_in_chunk;
    wire result_class;
    wire chunk_complete;
    wire fall_alert;
    wire [4:0] window_count_debug;
    wire [5:0] high_count_debug;

    integer index;

    gru_chunk_decision dut (
        .clk(clk),
        .reset(reset),
        .window_done(window_done),
        .probability_q15(probability_q15),
        .packet_motion_sum(packet_motion_sum),
        .packet_motion_pairs(packet_motion_pairs),
        .first_window_in_chunk(first_window_in_chunk),
        .result_class(result_class),
        .chunk_complete(chunk_complete),
        .fall_alert(fall_alert),
        .window_count_debug(window_count_debug),
        .high_count_debug(high_count_debug)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task apply_window;
        input [15:0] probability;
        input [39:0] motion_sum;
        input [15:0] motion_pairs;
        begin
            @(negedge clk);
            probability_q15 = probability;
            packet_motion_sum = motion_sum;
            packet_motion_pairs = motion_pairs;
            window_done = 1'b1;
            @(negedge clk);
            window_done = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            @(negedge clk);
            reset = 1'b1;
            repeat (3) @(negedge clk);
            reset = 1'b0;
        end
    endtask

    initial begin
        reset = 1'b1;
        window_done = 1'b0;
        probability_q15 = 0;
        packet_motion_sum = 0;
        packet_motion_pairs = 0;
        repeat (5) @(negedge clk);
        reset = 1'b0;

        apply_window(16'd26000, 40'd13000, 16'd10);
        if (result_class || fall_alert || chunk_complete) begin
            $display("CHUNK TEST FAILED: early alert");
            $finish;
        end

        apply_window(16'd26000, 40'd13000, 16'd10);
        for (index = 2; index < 17; index = index + 1)
            apply_window(16'd16500, 40'd13000, 16'd10);

        if (!fall_alert || !chunk_complete) begin
            $display("CHUNK TEST FAILED: tuned fall was not confirmed");
            $finish;
        end

        apply_reset();

        // Average and motion pass, but only one window reaches 0.70.
        apply_window(16'd26000, 40'd13000, 16'd10);
        for (index = 1; index < 17; index = index + 1)
            apply_window(16'd16500, 40'd13000, 16'd10);

        if (fall_alert) begin
            $display("CHUNK TEST FAILED: insufficient high-window false alert");
            $finish;
        end

        apply_reset();

        // Two high windows exist, but the average stays below 0.50.
        for (index = 0; index < 2; index = index + 1)
            apply_window(16'd26000, 40'd13000, 16'd10);
        for (index = 2; index < 17; index = index + 1)
            apply_window(16'd10000, 40'd13000, 16'd10);

        if (fall_alert) begin
            $display("CHUNK TEST FAILED: low average false alert");
            $finish;
        end

        apply_reset();

        // Probability passes, but motion is below 0.07.
        for (index = 0; index < 2; index = index + 1)
            apply_window(16'd26000, 40'd10000, 16'd10);
        for (index = 2; index < 17; index = index + 1)
            apply_window(16'd16500, 40'd10000, 16'd10);

        if (fall_alert) begin
            $display("CHUNK TEST FAILED: low motion false alert");
            $finish;
        end

        $display("GRU_MODERATE_CHUNK_DECISION_TEST PASSED");
        $finish;
    end

endmodule
