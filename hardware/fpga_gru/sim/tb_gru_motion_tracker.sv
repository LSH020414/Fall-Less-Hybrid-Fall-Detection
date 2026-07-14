`timescale 1ns / 1ps

module tb_gru_motion_tracker;

    reg clk;
    reg reset;
    reg input_we;
    reg [9:0] input_addr;
    reg signed [17:0] input_data;
    reg first_window_in_chunk;

    wire [39:0] packet_motion_sum;
    wire [15:0] packet_motion_pairs;

    integer address;
    integer joint;
    integer frame;

    gru_motion_tracker dut (
        .clk(clk),
        .reset(reset),
        .input_we(input_we),
        .input_addr(input_addr),
        .input_data(input_data),
        .first_window_in_chunk(first_window_in_chunk),
        .packet_motion_sum(packet_motion_sum),
        .packet_motion_pairs(packet_motion_pairs)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task write_feature;
        input [9:0] address_value;
        input signed [17:0] data_value;
        begin
            @(negedge clk);
            input_addr = address_value;
            input_data = data_value;
            input_we = 1'b1;
            @(negedge clk);
            input_we = 1'b0;
            repeat (7) @(negedge clk);
        end
    endtask

    initial begin
        reset = 1'b1;
        input_we = 1'b0;
        input_addr = 0;
        input_data = 0;
        first_window_in_chunk = 1'b1;
        repeat (5) @(negedge clk);
        reset = 1'b0;

        address = 0;
        for (frame = 0; frame < 20; frame = frame + 1) begin
            for (joint = 0; joint < 17; joint = joint + 1) begin
                write_feature(address[9:0], frame * 4);
                address = address + 1;
                write_feature(address[9:0], 18'sd0);
                address = address + 1;
                write_feature(address[9:0], 18'sd64);
                address = address + 1;
            end
        end

        repeat (20) @(negedge clk);

        if (packet_motion_pairs !== 16'd19 ||
            packet_motion_sum < 40'd19000 ||
            packet_motion_sum > 40'd19500) begin
            $display(
                "MOTION TEST FAILED pairs=%0d sum=%0d",
                packet_motion_pairs,
                packet_motion_sum
            );
            $finish;
        end

        $display(
            "GRU_MOTION_TRACKER_TEST PASSED pairs=%0d sum=%0d",
            packet_motion_pairs,
            packet_motion_sum
        );
        $finish;
    end

endmodule
