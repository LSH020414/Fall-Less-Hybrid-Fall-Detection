`timescale 1ns / 1ps

module tb_gru_fall_detector_top;

    localparam integer CLKS_PER_BIT = 4;
    localparam integer INPUT_COUNT = 1020;
    localparam integer SELECTED_CASE = 1;

    reg clk;
    reg reset;
    reg uart_rx_i;

    wire uart_tx_o;
    wire led;
    wire [6:0] seg;
    wire [3:0] an;
    wire dp;

    wire [7:0] monitor_rx_byte;
    wire monitor_rx_valid;
    reg signed [17:0] test_inputs [0:(5*INPUT_COUNT)-1];
    reg [7:0] response [0:6];

    integer feature_index;
    integer response_index;
    integer timeout_cycles;
    reg [7:0] checksum;
    reg [17:0] packed_feature;

    gru_fall_detector_top #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) dut (
        .clk(clk),
        .reset(reset),
        .uart_rx_i(uart_rx_i),
        .uart_tx_o(uart_tx_o),
        .led(led),
        .seg(seg),
        .an(an),
        .dp(dp)
    );

    uart_rx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_monitor_uart_rx (
        .clk(clk),
        .reset(reset),
        .rx(uart_tx_o),
        .rx_byte(monitor_rx_byte),
        .rx_valid(monitor_rx_valid)
    );

    initial begin
        $readmemh("sim/generated/gru_test_inputs.mem", test_inputs);
    end

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (monitor_rx_valid && response_index < 7) begin
            response[response_index] <= monitor_rx_byte;
            response_index <= response_index + 1;
        end
    end

    task send_byte;
        input [7:0] value;
        integer bit_index;
        begin
            @(negedge clk);
            uart_rx_i = 1'b0;
            repeat (CLKS_PER_BIT) @(negedge clk);
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                uart_rx_i = value[bit_index];
                repeat (CLKS_PER_BIT) @(negedge clk);
            end
            uart_rx_i = 1'b1;
            repeat (CLKS_PER_BIT) @(negedge clk);
        end
    endtask

    task send_payload_byte;
        input [7:0] value;
        begin
            checksum = checksum ^ value;
            send_byte(value);
        end
    endtask

    initial begin
        reset = 1'b1;
        uart_rx_i = 1'b1;
        response_index = 0;
        checksum = 0;

        repeat (10) @(posedge clk);
        reset = 1'b0;
        repeat (5) @(posedge clk);

        send_byte(8'hA5);
        send_byte(8'h5A);
        send_payload_byte(8'h34);
        send_payload_byte(8'h12);

        for (feature_index = 0; feature_index < INPUT_COUNT; feature_index = feature_index + 1) begin
            packed_feature = test_inputs[(SELECTED_CASE * INPUT_COUNT) + feature_index];
            send_payload_byte(packed_feature[7:0]);
            send_payload_byte(packed_feature[15:8]);
            send_payload_byte({6'd0, packed_feature[17:16]});
        end
        send_byte(checksum);

        timeout_cycles = 0;
        while ((response_index < 7) && (timeout_cycles < 1000000)) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end

        if (response_index != 7) begin
            $display("TOP TEST FAILED: UART response timeout bytes=%0d", response_index);
            $finish;
        end

        if (response[0] !== 8'hF1 ||
            response[1] !== 8'h34 ||
            response[2] !== 8'h12 ||
            {response[4], response[3]} !== 16'd32757 ||
            response[5] !== 8'h00 ||
            response[6] !== (response[0] ^ response[1] ^ response[2] ^
                             response[3] ^ response[4] ^ response[5]) ||
            led !== 1'b0) begin
            $display(
                "TOP TEST FAILED response=%02x %02x %02x %02x %02x %02x %02x led=%b",
                response[0], response[1], response[2], response[3],
                response[4], response[5], response[6], led
            );
            $finish;
        end

        $display(
            "GRU_UART_TOP_TEST PASSED single_window_rejected probability_q15=%0d response=%02x %02x %02x %02x %02x %02x %02x",
            {response[4], response[3]},
            response[0], response[1], response[2], response[3],
            response[4], response[5], response[6]
        );
        $finish;
    end

endmodule
