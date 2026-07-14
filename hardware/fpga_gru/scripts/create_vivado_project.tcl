set project_dir [file normalize [file join [file dirname [info script]] ".."]]
set project_file [file join $project_dir "GRU_Fall_Detect.xpr"]

create_project GRU_Fall_Detect $project_dir -part xc7a35tcpg236-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

add_files -norecurse [list \
    [file join $project_dir "src" "uart_rx.v"] \
    [file join $project_dir "src" "uart_tx.v"] \
    [file join $project_dir "src" "seven_segment_fall_display.v"] \
    [file join $project_dir "src" "gru_sequence_uart_parser.sv"] \
    [file join $project_dir "src" "gru_result_uart_tx.sv"] \
    [file join $project_dir "src" "gru_fixed_core.sv"] \
    [file join $project_dir "src" "gru_motion_tracker.sv"] \
    [file join $project_dir "src" "gru_chunk_decision.sv"] \
    [file join $project_dir "src" "gru_fall_detector_top.sv"] \
]

set mem_files [glob -nocomplain [file join $project_dir "mem" "*.mem"]]
if {[llength $mem_files] > 0} {
    add_files -norecurse $mem_files
    set_property file_type {Memory Initialization Files} [get_files -of_objects [get_filesets sources_1] *.mem]
}

add_files -fileset constrs_1 -norecurse \
    [file join $project_dir "constraints" "basys3_gru_fall_detector.xdc"]

add_files -fileset sim_1 -norecurse [list \
    [file join $project_dir "sim" "tb_gru_fixed_core.sv"] \
    [file join $project_dir "sim" "tb_gru_motion_tracker.sv"] \
    [file join $project_dir "sim" "tb_gru_chunk_decision.sv"] \
    [file join $project_dir "sim" "tb_gru_fall_detector_top.sv"] \
]
set generated_files [glob -nocomplain [file join $project_dir "sim" "generated" "*.mem"]]
if {[llength $generated_files] > 0} {
    add_files -fileset sim_1 -norecurse $generated_files
}

set_property top gru_fall_detector_top [get_filesets sources_1]
set_property top tb_gru_fall_detector_top [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
close_project

puts "PROJECT_CREATED=$project_file"
