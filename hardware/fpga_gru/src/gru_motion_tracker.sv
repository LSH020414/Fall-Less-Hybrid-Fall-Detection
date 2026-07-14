`timescale 1ns / 1ps

module gru_motion_tracker (
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  input_we,
    input  wire [9:0]            input_addr,
    input  wire signed [17:0]    input_data,
    input  wire                  first_window_in_chunk,
    output reg  [39:0]           packet_motion_sum,
    output reg  [15:0]           packet_motion_pairs
);

    reg [1:0] feature_phase;
    reg [4:0] joint_index;
    reg [4:0] frame_index;
    reg signed [17:0] current_x;
    reg signed [17:0] current_y;

    reg signed [17:0] previous_x [0:16];
    reg signed [17:0] previous_y [0:16];
    reg signed [17:0] previous_confidence [0:16];

    reg [2:0] joint_pipeline_stage;
    reg joint_pending_valid;
    reg joint_pending_last;
    reg signed [17:0] joint_current_x;
    reg signed [17:0] joint_current_y;
    reg signed [17:0] joint_previous_x;
    reg signed [17:0] joint_previous_y;
    reg [18:0] joint_dx;
    reg [18:0] joint_dy;
    reg [18:0] joint_max;
    reg [18:0] joint_min;
    reg [20:0] joint_distance;

    reg [23:0] frame_distance_sum;
    reg [4:0] frame_valid_count;
    reg [24:0] pending_frame_distance;
    reg [4:0] pending_frame_valid_count;
    reg pending_frame_valid;
    reg [33:0] pending_frame_mean_q14;
    reg pending_mean_valid;

    integer index;

    function automatic [18:0] absolute_difference;
        input signed [17:0] left;
        input signed [17:0] right;
        reg signed [18:0] difference;
        begin
            difference = {left[17], left} - {right[17], right};
            absolute_difference =
                difference < 0 ? -difference : difference;
        end
    endfunction

    function automatic [8:0] reciprocal_q8;
        input [4:0] count;
        begin
            case (count)
                1: reciprocal_q8 = 256;
                2: reciprocal_q8 = 128;
                3: reciprocal_q8 = 85;
                4: reciprocal_q8 = 64;
                5: reciprocal_q8 = 51;
                6: reciprocal_q8 = 43;
                7: reciprocal_q8 = 37;
                8: reciprocal_q8 = 32;
                9: reciprocal_q8 = 28;
                10: reciprocal_q8 = 26;
                11: reciprocal_q8 = 23;
                12: reciprocal_q8 = 21;
                13: reciprocal_q8 = 20;
                14: reciprocal_q8 = 18;
                15: reciprocal_q8 = 17;
                16: reciprocal_q8 = 16;
                17: reciprocal_q8 = 15;
                default: reciprocal_q8 = 0;
            endcase
        end
    endfunction

    wire transition_selected =
        (frame_index != 0) &&
        (first_window_in_chunk || (frame_index >= 15));

    always @(posedge clk) begin
        if (reset) begin
            feature_phase <= 0;
            joint_index <= 0;
            frame_index <= 0;
            current_x <= 0;
            current_y <= 0;
            joint_pipeline_stage <= 0;
            joint_pending_valid <= 0;
            joint_pending_last <= 0;
            joint_current_x <= 0;
            joint_current_y <= 0;
            joint_previous_x <= 0;
            joint_previous_y <= 0;
            joint_dx <= 0;
            joint_dy <= 0;
            joint_max <= 0;
            joint_min <= 0;
            joint_distance <= 0;
            frame_distance_sum <= 0;
            frame_valid_count <= 0;
            pending_frame_distance <= 0;
            pending_frame_valid_count <= 0;
            pending_frame_valid <= 0;
            pending_frame_mean_q14 <= 0;
            pending_mean_valid <= 0;
            packet_motion_sum <= 0;
            packet_motion_pairs <= 0;
            for (index = 0; index < 17; index = index + 1) begin
                previous_x[index] <= 0;
                previous_y[index] <= 0;
                previous_confidence[index] <= 0;
            end
        end else begin
            case (joint_pipeline_stage)
                1: begin
                    joint_dx <= absolute_difference(
                        joint_current_x,
                        joint_previous_x
                    );
                    joint_dy <= absolute_difference(
                        joint_current_y,
                        joint_previous_y
                    );
                    joint_pipeline_stage <= 2;
                end

                2: begin
                    if (joint_dx >= joint_dy) begin
                        joint_max <= joint_dx;
                        joint_min <= joint_dy;
                    end else begin
                        joint_max <= joint_dy;
                        joint_min <= joint_dx;
                    end
                    joint_pipeline_stage <= 3;
                end

                3: begin
                    joint_distance <=
                        joint_max + (joint_min >> 2) + (joint_min >> 3);
                    joint_pipeline_stage <= 4;
                end

                4: begin
                    if (joint_pending_last) begin
                        if (joint_pending_valid) begin
                            pending_frame_distance <=
                                frame_distance_sum + joint_distance;
                            pending_frame_valid_count <=
                                frame_valid_count + 1'b1;
                        end else begin
                            pending_frame_distance <= frame_distance_sum;
                            pending_frame_valid_count <= frame_valid_count;
                        end

                        if ((frame_valid_count != 0) || joint_pending_valid)
                            pending_frame_valid <= 1;

                        frame_distance_sum <= 0;
                        frame_valid_count <= 0;
                    end else if (joint_pending_valid) begin
                        frame_distance_sum <=
                            frame_distance_sum + joint_distance;
                        frame_valid_count <= frame_valid_count + 1'b1;
                    end
                    joint_pipeline_stage <= 0;
                end

                default: joint_pipeline_stage <= 0;
            endcase

            if (pending_frame_valid) begin
                pending_frame_mean_q14 <=
                    pending_frame_distance *
                    reciprocal_q8(pending_frame_valid_count);
                pending_frame_valid <= 0;
                pending_mean_valid <= 1;
            end else if (pending_mean_valid) begin
                packet_motion_sum <=
                    packet_motion_sum + pending_frame_mean_q14;
                packet_motion_pairs <= packet_motion_pairs + 1'b1;
                pending_mean_valid <= 0;
            end

            if (input_we && (input_addr == 0)) begin
                feature_phase <= 1;
                joint_index <= 0;
                frame_index <= 0;
                current_x <= input_data;
                current_y <= 0;
                joint_pipeline_stage <= 0;
                frame_distance_sum <= 0;
                frame_valid_count <= 0;
                packet_motion_sum <= 0;
                packet_motion_pairs <= 0;
                pending_frame_valid <= 0;
                pending_mean_valid <= 0;
            end else if (input_we) begin
                case (feature_phase)
                    0: begin
                        current_x <= input_data;
                        feature_phase <= 1;
                    end

                    1: begin
                        current_y <= input_data;
                        feature_phase <= 2;
                    end

                    2: begin
                        joint_pending_valid <=
                            transition_selected &&
                            (previous_confidence[joint_index] > 0) &&
                            (input_data > 0);
                        joint_pending_last <= joint_index == 16;
                        joint_current_x <= current_x;
                        joint_current_y <= current_y;
                        joint_previous_x <= previous_x[joint_index];
                        joint_previous_y <= previous_y[joint_index];
                        joint_pipeline_stage <= 1;

                        previous_x[joint_index] <= current_x;
                        previous_y[joint_index] <= current_y;
                        previous_confidence[joint_index] <= input_data;
                        feature_phase <= 0;

                        if (joint_index == 16) begin
                            joint_index <= 0;
                            frame_index <= frame_index + 1'b1;
                        end else begin
                            joint_index <= joint_index + 1'b1;
                        end
                    end

                    default: feature_phase <= 0;
                endcase
            end
        end
    end

endmodule
