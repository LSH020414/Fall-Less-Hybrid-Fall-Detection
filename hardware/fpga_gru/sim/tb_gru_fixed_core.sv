`timescale 1ns / 1ps

module tb_gru_fixed_core;

    localparam integer CASES = 5;
    localparam integer INPUT_COUNT = 20 * 51;

    reg clk;
    reg reset;
    reg input_we;
    reg [9:0] input_addr;
    reg signed [17:0] input_data;
    reg start;

    wire busy;
    wire done;
    wire [15:0] fall_probability_q15;
    wire signed [31:0] logit0_q12;
    wire signed [31:0] logit1_q12;

    reg signed [17:0] test_inputs [0:(CASES*INPUT_COUNT)-1];
    reg [15:0] expected_probability [0:CASES-1];
    reg signed [31:0] expected_logits [0:(CASES*2)-1];

    integer case_index;
    integer feature_index;
    integer timeout_cycles;

    gru_fixed_core dut (
        .clk(clk),
        .reset(reset),
        .input_we(input_we),
        .input_addr(input_addr),
        .input_data(input_data),
        .start(start),
        .busy(busy),
        .done(done),
        .fall_probability_q15(fall_probability_q15),
        .logit0_q12(logit0_q12),
        .logit1_q12(logit1_q12)
    );

    initial begin
        $readmemh("sim/generated/gru_test_inputs.mem", test_inputs);
        $readmemh("sim/generated/gru_test_expected_probability.mem", expected_probability);
        $readmemh("sim/generated/gru_test_expected_logits.mem", expected_logits);
    end

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task load_case;
        input integer selected_case;
        begin
            for (feature_index = 0; feature_index < INPUT_COUNT; feature_index = feature_index + 1) begin
                @(negedge clk);
                input_we = 1'b1;
                input_addr = feature_index[9:0];
                input_data = test_inputs[(selected_case * INPUT_COUNT) + feature_index];
            end
            @(negedge clk);
            input_we = 1'b0;
            input_addr = 10'd0;
            input_data = 18'sd0;
        end
    endtask

    task run_case;
        input integer selected_case;
        begin
            load_case(selected_case);
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            timeout_cycles = 0;
            while (!done && timeout_cycles < 1500000) begin
                @(posedge clk);
                timeout_cycles = timeout_cycles + 1;
            end

            if (!done) begin
                $display("TEST FAILED: timeout case=%0d state=%0d", selected_case, dut.state);
                $finish;
            end

            if (fall_probability_q15 !== expected_probability[selected_case] ||
                logit0_q12 !== expected_logits[selected_case * 2] ||
                logit1_q12 !== expected_logits[(selected_case * 2) + 1]) begin
                $display(
                    "TEST FAILED case=%0d cycles=%0d prob actual=%0d expected=%0d logit0 actual=%0d expected=%0d logit1 actual=%0d expected=%0d",
                    selected_case,
                    timeout_cycles,
                    fall_probability_q15,
                    expected_probability[selected_case],
                    logit0_q12,
                    expected_logits[selected_case * 2],
                    logit1_q12,
                    expected_logits[(selected_case * 2) + 1]
                );
                $finish;
            end

            $display(
                "GRU_CASE_PASS case=%0d cycles=%0d probability_q15=%0d logits=(%0d,%0d)",
                selected_case,
                timeout_cycles,
                fall_probability_q15,
                logit0_q12,
                logit1_q12
            );
            repeat (3) @(posedge clk);
        end
    endtask

    initial begin
        reset = 1'b1;
        input_we = 1'b0;
        input_addr = 10'd0;
        input_data = 18'sd0;
        start = 1'b0;

        repeat (5) @(posedge clk);
        reset = 1'b0;
        repeat (3) @(posedge clk);

        for (case_index = 0; case_index < CASES; case_index = case_index + 1)
            run_case(case_index);

        $display("GRU_FIXED_CORE_TEST PASSED cases=%0d", CASES);
        $finish;
    end

endmodule
