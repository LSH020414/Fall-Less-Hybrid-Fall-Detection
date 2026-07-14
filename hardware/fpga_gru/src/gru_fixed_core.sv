`timescale 1ns / 1ps

module gru_fixed_core #(
    parameter integer INPUT_SIZE = 51,
    parameter integer HIDDEN_SIZE = 64,
    parameter integer SEQ_LEN = 20,
    parameter integer INPUT_BITS = 18,
    parameter integer WEIGHT_BITS = 16,
    parameter integer HIDDEN_BITS = 16,
    parameter integer PRE_BITS = 32,
    parameter integer INPUT_WEIGHT_SHIFT = 8,
    parameter integer HIDDEN_WEIGHT_SHIFT = 16,
    parameter integer GATE_FRAC = 15,
    parameter integer LUT_STEP_SHIFT = 6,
    parameter integer LUT_X_MAX_Q = 32768,
    parameter integer LUT_SIZE = 1025,
    parameter W_IH_FILE = "mem/gru_weight_ih_l0.mem",
    parameter W_HH_FILE = "mem/gru_weight_hh_l0.mem",
    parameter B_IH_FILE = "mem/gru_bias_ih_l0.mem",
    parameter B_HH_FILE = "mem/gru_bias_hh_l0.mem",
    parameter FC_W_FILE = "mem/fc_weight.mem",
    parameter FC_B_FILE = "mem/fc_bias.mem",
    parameter SIGMOID_FILE = "mem/sigmoid_lut.mem",
    parameter TANH_FILE = "mem/tanh_lut.mem"
) (
    input  wire                         clk,
    input  wire                         reset,
    input  wire                         input_we,
    input  wire [9:0]                   input_addr,
    input  wire signed [INPUT_BITS-1:0] input_data,
    input  wire                         start,
    output reg                          busy,
    output reg                          done,
    output reg [15:0]                   fall_probability_q15,
    output reg signed [PRE_BITS-1:0]    logit0_q12,
    output reg signed [PRE_BITS-1:0]    logit1_q12
);

    localparam integer INPUT_COUNT = INPUT_SIZE * SEQ_LEN;
    localparam integer W_IH_COUNT = 3 * HIDDEN_SIZE * INPUT_SIZE;
    localparam integer W_HH_COUNT = 3 * HIDDEN_SIZE * HIDDEN_SIZE;
    localparam integer BIAS_COUNT = 3 * HIDDEN_SIZE;
    localparam integer FC_WEIGHT_COUNT = 2 * HIDDEN_SIZE;
    localparam signed [16:0] GATE_ONE = 17'sd32768;

    localparam [5:0]
        S_IDLE       = 6'd0,
        S_RX_X_SETUP = 6'd1,
        S_MAC        = 6'd2,
        S_RX_H_SETUP = 6'd3,
        S_R_PRE      = 6'd4,
        S_R_ACT      = 6'd5,
        S_ZX_SETUP   = 6'd6,
        S_ZH_SETUP   = 6'd7,
        S_Z_PRE      = 6'd8,
        S_Z_ACT      = 6'd9,
        S_NX_SETUP   = 6'd10,
        S_NH_SETUP   = 6'd11,
        S_N_PRE      = 6'd12,
        S_N_MUL      = 6'd13,
        S_N_ACT      = 6'd14,
        S_H_MIX      = 6'd15,
        S_H_UPDATE   = 6'd16,
        S_FC0_SETUP  = 6'd17,
        S_FC1_SETUP  = 6'd18,
        S_OUT_LOGITS = 6'd19,
        S_OUTPUT     = 6'd20,
        S_N_ROUND    = 6'd21,
        S_H_READ     = 6'd22,
        S_H_ROUND    = 6'd23;

    localparam [2:0]
        MAC_RX_X = 3'd0,
        MAC_RX_H = 3'd1,
        MAC_Z_X  = 3'd2,
        MAC_Z_H  = 3'd3,
        MAC_N_X  = 3'd4,
        MAC_N_H  = 3'd5,
        MAC_FC0  = 3'd6,
        MAC_FC1  = 3'd7;

    reg signed [INPUT_BITS-1:0] input_mem [0:INPUT_COUNT-1];
    (* rom_style = "block" *)
    reg signed [WEIGHT_BITS-1:0] w_ih_mem [0:W_IH_COUNT-1];
    (* rom_style = "block" *)
    reg signed [WEIGHT_BITS-1:0] w_hh_mem [0:W_HH_COUNT-1];
    reg signed [PRE_BITS-1:0] b_ih_mem [0:BIAS_COUNT-1];
    reg signed [PRE_BITS-1:0] b_hh_mem [0:BIAS_COUNT-1];
    (* rom_style = "block" *)
    reg signed [WEIGHT_BITS-1:0] fc_w_mem [0:FC_WEIGHT_COUNT-1];
    reg signed [PRE_BITS-1:0] fc_b_mem [0:1];
    reg [15:0] sigmoid_mem [0:LUT_SIZE-1];
    reg signed [15:0] tanh_mem [0:LUT_SIZE-1];

    reg signed [HIDDEN_BITS-1:0] hidden_a [0:HIDDEN_SIZE-1];
    reg signed [HIDDEN_BITS-1:0] hidden_b [0:HIDDEN_SIZE-1];

    reg [5:0] state;
    reg [2:0] mac_kind;
    reg [4:0] time_index;
    reg [5:0] unit_index;
    reg hidden_bank;

    reg [13:0] mac_base;
    reg [6:0] mac_length;
    reg [6:0] mac_issue_count;
    reg [6:0] mac_consume_count;
    reg [13:0] w_ih_read_addr;
    reg [13:0] w_hh_read_addr;
    reg [6:0] fc_w_read_addr;
    reg signed [WEIGHT_BITS-1:0] w_ih_read_data;
    reg signed [WEIGHT_BITS-1:0] w_hh_read_data;
    reg signed [WEIGHT_BITS-1:0] fc_w_read_data;
    reg mac_issue_valid_d1;
    reg mac_issue_valid_d2;
    reg [1:0] mac_source_d1;
    reg [1:0] mac_source_d2;
    reg signed [INPUT_BITS-1:0] mac_operand_d1;
    reg signed [INPUT_BITS-1:0] mac_operand_d2;
    reg signed [47:0] mac_accumulator;

    reg signed [47:0] x_accumulator;
    reg signed [47:0] h_accumulator;
    reg [15:0] reset_gate_q15;
    reg [15:0] update_gate_q15;
    reg signed [15:0] new_gate_q14;
    reg signed [47:0] fc0_accumulator;
    reg signed [47:0] fc1_accumulator;
    reg signed [PRE_BITS-1:0] preactivation_reg;
    reg signed [PRE_BITS-1:0] n_input_q12_reg;
    reg signed [PRE_BITS-1:0] n_hidden_q12_reg;
    reg signed [PRE_BITS-1:0] n_product_q12_reg;
    reg signed [47:0] n_product_raw_reg;
    reg signed [HIDDEN_BITS-1:0] hidden_current_reg;
    reg signed [32:0] hidden_mix_reg;
    reg signed [32:0] hidden_rounded_reg;
    reg signed [PRE_BITS-1:0] output_logit0_reg;
    reg signed [PRE_BITS-1:0] output_logit1_reg;

    integer i;
    integer issue_address;
    reg signed [HIDDEN_BITS-1:0] issue_hidden;
    reg signed [WEIGHT_BITS-1:0] selected_weight;
    reg signed [33:0] current_product;
    reg signed [47:0] final_accumulator;
    reg signed [PRE_BITS-1:0] preactivation;
    reg signed [HIDDEN_BITS-1:0] updated_hidden;
    reg signed [PRE_BITS-1:0] output_difference;

    initial begin
        $readmemh(W_IH_FILE, w_ih_mem);
        $readmemh(W_HH_FILE, w_hh_mem);
        $readmemh(B_IH_FILE, b_ih_mem);
        $readmemh(B_HH_FILE, b_hh_mem);
        $readmemh(FC_W_FILE, fc_w_mem);
        $readmemh(FC_B_FILE, fc_b_mem);
        $readmemh(SIGMOID_FILE, sigmoid_mem);
        $readmemh(TANH_FILE, tanh_mem);
    end

    always @(posedge clk) begin
        w_ih_read_data <= w_ih_mem[w_ih_read_addr];
        w_hh_read_data <= w_hh_mem[w_hh_read_addr];
        fc_w_read_data <= fc_w_mem[fc_w_read_addr];
    end

    function automatic signed [47:0] round_shift48;
        input signed [47:0] value;
        input integer shift;
        reg signed [47:0] magnitude;
        begin
            if (shift == 0) begin
                round_shift48 = value;
            end else if (value >= 0) begin
                round_shift48 = (value + (48'sd1 <<< (shift - 1))) >>> shift;
            end else begin
                magnitude = -value;
                round_shift48 = -((magnitude + (48'sd1 <<< (shift - 1))) >>> shift);
            end
        end
    endfunction

    function automatic signed [32:0] round_shift33;
        input signed [32:0] value;
        input integer shift;
        reg signed [32:0] magnitude;
        begin
            if (value >= 0) begin
                round_shift33 = (value + (33'sd1 <<< (shift - 1))) >>> shift;
            end else begin
                magnitude = -value;
                round_shift33 = -((magnitude + (33'sd1 <<< (shift - 1))) >>> shift);
            end
        end
    endfunction

    function automatic [10:0] activation_index;
        input signed [PRE_BITS-1:0] value;
        reg signed [PRE_BITS-1:0] clipped;
        reg signed [PRE_BITS:0] shifted;
        begin
            if (value < -LUT_X_MAX_Q)
                clipped = -LUT_X_MAX_Q;
            else if (value > LUT_X_MAX_Q)
                clipped = LUT_X_MAX_Q;
            else
                clipped = value;
            shifted = clipped + LUT_X_MAX_Q + (1 <<< (LUT_STEP_SHIFT - 1));
            activation_index = shifted >>> LUT_STEP_SHIFT;
        end
    endfunction

    function automatic signed [HIDDEN_BITS-1:0] current_hidden;
        input [5:0] index;
        begin
            current_hidden = hidden_bank ? hidden_b[index] : hidden_a[index];
        end
    endfunction

    function automatic signed [HIDDEN_BITS-1:0] saturate_hidden;
        input signed [32:0] value;
        begin
            if (value > 32767)
                saturate_hidden = 16'sh7fff;
            else if (value < -32768)
                saturate_hidden = 16'sh8000;
            else
                saturate_hidden = value[HIDDEN_BITS-1:0];
        end
    endfunction

    always @(posedge clk) begin
        if (input_we && !busy && (input_addr < INPUT_COUNT))
            input_mem[input_addr] <= input_data;

        if (reset) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            fall_probability_q15 <= 16'd0;
            logit0_q12 <= {PRE_BITS{1'b0}};
            logit1_q12 <= {PRE_BITS{1'b0}};
            mac_issue_valid_d1 <= 1'b0;
            mac_issue_valid_d2 <= 1'b0;
            mac_accumulator <= 48'sd0;
            hidden_bank <= 1'b0;
            hidden_rounded_reg <= 33'sd0;
            for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                hidden_a[i] <= {HIDDEN_BITS{1'b0}};
                hidden_b[i] <= {HIDDEN_BITS{1'b0}};
            end
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        time_index <= 5'd0;
                        unit_index <= 6'd0;
                        hidden_bank <= 1'b0;
                        mac_issue_valid_d1 <= 1'b0;
                        mac_issue_valid_d2 <= 1'b0;
                        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                            hidden_a[i] <= {HIDDEN_BITS{1'b0}};
                            hidden_b[i] <= {HIDDEN_BITS{1'b0}};
                        end
                        state <= S_RX_X_SETUP;
                    end
                end

                S_RX_X_SETUP: begin
                    mac_kind <= MAC_RX_X;
                    mac_base <= unit_index * INPUT_SIZE;
                    mac_length <= INPUT_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_RX_H_SETUP: begin
                    mac_kind <= MAC_RX_H;
                    mac_base <= unit_index * HIDDEN_SIZE;
                    mac_length <= HIDDEN_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_ZX_SETUP: begin
                    mac_kind <= MAC_Z_X;
                    mac_base <= (HIDDEN_SIZE + unit_index) * INPUT_SIZE;
                    mac_length <= INPUT_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_ZH_SETUP: begin
                    mac_kind <= MAC_Z_H;
                    mac_base <= (HIDDEN_SIZE + unit_index) * HIDDEN_SIZE;
                    mac_length <= HIDDEN_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_NX_SETUP: begin
                    mac_kind <= MAC_N_X;
                    mac_base <= ((2 * HIDDEN_SIZE) + unit_index) * INPUT_SIZE;
                    mac_length <= INPUT_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_NH_SETUP: begin
                    mac_kind <= MAC_N_H;
                    mac_base <= ((2 * HIDDEN_SIZE) + unit_index) * HIDDEN_SIZE;
                    mac_length <= HIDDEN_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_FC0_SETUP: begin
                    mac_kind <= MAC_FC0;
                    mac_base <= 0;
                    mac_length <= HIDDEN_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_FC1_SETUP: begin
                    mac_kind <= MAC_FC1;
                    mac_base <= HIDDEN_SIZE;
                    mac_length <= HIDDEN_SIZE;
                    mac_issue_count <= 0;
                    mac_consume_count <= 0;
                    mac_issue_valid_d1 <= 1'b0;
                    mac_issue_valid_d2 <= 1'b0;
                    mac_accumulator <= 48'sd0;
                    state <= S_MAC;
                end

                S_MAC: begin
                    mac_issue_valid_d2 <= mac_issue_valid_d1;
                    mac_source_d2 <= mac_source_d1;
                    mac_operand_d2 <= mac_operand_d1;

                    if (mac_issue_count < mac_length) begin
                        issue_address = mac_base + mac_issue_count;
                        issue_hidden = current_hidden(mac_issue_count[5:0]);
                        case (mac_kind)
                            MAC_RX_X, MAC_Z_X, MAC_N_X: begin
                                w_ih_read_addr <= issue_address[13:0];
                                mac_source_d1 <= 2'd0;
                                mac_operand_d1 <= input_mem[(time_index * INPUT_SIZE) + mac_issue_count];
                            end
                            MAC_RX_H, MAC_Z_H, MAC_N_H: begin
                                w_hh_read_addr <= issue_address[13:0];
                                mac_source_d1 <= 2'd1;
                                mac_operand_d1 <= {{(INPUT_BITS-HIDDEN_BITS){issue_hidden[HIDDEN_BITS-1]}},
                                                   issue_hidden};
                            end
                            default: begin
                                fc_w_read_addr <= issue_address[6:0];
                                mac_source_d1 <= 2'd2;
                                mac_operand_d1 <= {{(INPUT_BITS-HIDDEN_BITS){issue_hidden[HIDDEN_BITS-1]}},
                                                   issue_hidden};
                            end
                        endcase
                        mac_issue_count <= mac_issue_count + 1'b1;
                        mac_issue_valid_d1 <= 1'b1;
                    end else begin
                        mac_issue_valid_d1 <= 1'b0;
                    end

                    if (mac_issue_valid_d2) begin
                        case (mac_source_d2)
                            2'd0: selected_weight = w_ih_read_data;
                            2'd1: selected_weight = w_hh_read_data;
                            default: selected_weight = fc_w_read_data;
                        endcase
                        current_product = $signed(selected_weight) * $signed(mac_operand_d2);
                        final_accumulator = mac_accumulator + current_product;
                        mac_accumulator <= final_accumulator;
                        mac_consume_count <= mac_consume_count + 1'b1;

                        if ((mac_consume_count + 1'b1) == mac_length) begin
                            mac_issue_valid_d1 <= 1'b0;
                            mac_issue_valid_d2 <= 1'b0;
                            case (mac_kind)
                                MAC_RX_X: begin
                                    x_accumulator <= final_accumulator;
                                    state <= S_RX_H_SETUP;
                                end
                                MAC_RX_H: begin
                                    h_accumulator <= final_accumulator;
                                    state <= S_R_PRE;
                                end
                                MAC_Z_X: begin
                                    x_accumulator <= final_accumulator;
                                    state <= S_ZH_SETUP;
                                end
                                MAC_Z_H: begin
                                    h_accumulator <= final_accumulator;
                                    state <= S_Z_PRE;
                                end
                                MAC_N_X: begin
                                    x_accumulator <= final_accumulator;
                                    state <= S_NH_SETUP;
                                end
                                MAC_N_H: begin
                                    h_accumulator <= final_accumulator;
                                    state <= S_N_PRE;
                                end
                                MAC_FC0: begin
                                    fc0_accumulator <= final_accumulator;
                                    state <= S_FC1_SETUP;
                                end
                                default: begin
                                    fc1_accumulator <= final_accumulator;
                                    state <= S_OUT_LOGITS;
                                end
                            endcase
                        end
                    end
                end

                S_R_PRE: begin
                    preactivation_reg <=
                        round_shift48(x_accumulator, INPUT_WEIGHT_SHIFT) +
                        b_ih_mem[unit_index] +
                        round_shift48(h_accumulator, HIDDEN_WEIGHT_SHIFT) +
                        b_hh_mem[unit_index];
                    state <= S_R_ACT;
                end

                S_R_ACT: begin
                    reset_gate_q15 <= sigmoid_mem[activation_index(preactivation_reg)];
                    state <= S_ZX_SETUP;
                end

                S_Z_PRE: begin
                    preactivation_reg <=
                        round_shift48(x_accumulator, INPUT_WEIGHT_SHIFT) +
                        b_ih_mem[HIDDEN_SIZE + unit_index] +
                        round_shift48(h_accumulator, HIDDEN_WEIGHT_SHIFT) +
                        b_hh_mem[HIDDEN_SIZE + unit_index];
                    state <= S_Z_ACT;
                end

                S_Z_ACT: begin
                    update_gate_q15 <= sigmoid_mem[activation_index(preactivation_reg)];
                    state <= S_NX_SETUP;
                end

                S_N_PRE: begin
                    n_input_q12_reg <=
                        round_shift48(x_accumulator, INPUT_WEIGHT_SHIFT) +
                        b_ih_mem[(2 * HIDDEN_SIZE) + unit_index];
                    n_hidden_q12_reg <=
                        round_shift48(h_accumulator, HIDDEN_WEIGHT_SHIFT) +
                        b_hh_mem[(2 * HIDDEN_SIZE) + unit_index];
                    state <= S_N_MUL;
                end

                S_N_MUL: begin
                    n_product_raw_reg <=
                        $signed({1'b0, reset_gate_q15}) * n_hidden_q12_reg;
                    state <= S_N_ROUND;
                end

                S_N_ROUND: begin
                    n_product_q12_reg <=
                        round_shift48(n_product_raw_reg, GATE_FRAC);
                    state <= S_N_ACT;
                end

                S_N_ACT: begin
                    preactivation = n_input_q12_reg + n_product_q12_reg;
                    new_gate_q14 <= tanh_mem[activation_index(preactivation)];
                    state <= S_H_READ;
                end

                S_H_READ: begin
                    hidden_current_reg <= current_hidden(unit_index);
                    state <= S_H_MIX;
                end

                S_H_MIX: begin
                    hidden_mix_reg <=
                        $signed({1'b0, update_gate_q15}) * hidden_current_reg +
                        $signed(GATE_ONE - $signed({1'b0, update_gate_q15})) *
                        $signed(new_gate_q14);
                    state <= S_H_ROUND;
                end

                S_H_ROUND: begin
                    hidden_rounded_reg <=
                        round_shift33(hidden_mix_reg, GATE_FRAC);
                    state <= S_H_UPDATE;
                end

                S_H_UPDATE: begin
                    updated_hidden = saturate_hidden(hidden_rounded_reg);

                    if (hidden_bank)
                        hidden_a[unit_index] <= updated_hidden;
                    else
                        hidden_b[unit_index] <= updated_hidden;

                    if (unit_index == HIDDEN_SIZE - 1) begin
                        unit_index <= 0;
                        hidden_bank <= ~hidden_bank;
                        if (time_index == SEQ_LEN - 1) begin
                            state <= S_FC0_SETUP;
                        end else begin
                            time_index <= time_index + 1'b1;
                            state <= S_RX_X_SETUP;
                        end
                    end else begin
                        unit_index <= unit_index + 1'b1;
                        state <= S_RX_X_SETUP;
                    end
                end

                S_OUT_LOGITS: begin
                    output_logit0_reg <=
                        round_shift48(fc0_accumulator, HIDDEN_WEIGHT_SHIFT) +
                        fc_b_mem[0];
                    output_logit1_reg <=
                        round_shift48(fc1_accumulator, HIDDEN_WEIGHT_SHIFT) +
                        fc_b_mem[1];
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    output_difference = output_logit1_reg - output_logit0_reg;
                    logit0_q12 <= output_logit0_reg;
                    logit1_q12 <= output_logit1_reg;
                    fall_probability_q15 <= sigmoid_mem[activation_index(output_difference)];
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
