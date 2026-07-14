set project_dir [file normalize [file join [file dirname [info script]] ".."]]
set output_dir [file join $project_dir "reports" "final_bitstream"]
file mkdir $output_dir
cd $project_dir

read_verilog [list \
    [file join $project_dir "src" "uart_rx.v"] \
    [file join $project_dir "src" "uart_tx.v"] \
    [file join $project_dir "src" "seven_segment_fall_display.v"] \
]
read_verilog -sv [list \
    [file join $project_dir "src" "gru_sequence_uart_parser.sv"] \
    [file join $project_dir "src" "gru_result_uart_tx.sv"] \
    [file join $project_dir "src" "gru_fixed_core.sv"] \
    [file join $project_dir "src" "gru_motion_tracker.sv"] \
    [file join $project_dir "src" "gru_chunk_decision.sv"] \
    [file join $project_dir "src" "gru_fall_detector_top.sv"] \
]
read_xdc [file join $project_dir "constraints" "basys3_gru_fall_detector.xdc"]

synth_design -top gru_fall_detector_top -part xc7a35tcpg236-1
opt_design
place_design
phys_opt_design
route_design -directive Explore

set routed_path [get_timing_paths -delay_type max -max_paths 1]
set routed_wns [get_property SLACK $routed_path]
puts "ROUTED_WNS=$routed_wns"
if {$routed_wns < 0.0} {
    puts "Running post-route physical optimization."
    phys_opt_design -directive AggressiveExplore
}

report_utilization -file [file join $output_dir "gru_impl_utilization.rpt"]
report_timing_summary -file [file join $output_dir "gru_impl_timing.rpt"]
report_drc -file [file join $output_dir "gru_impl_drc.rpt"]

set worst_path [get_timing_paths -delay_type max -max_paths 1]
set final_wns [get_property SLACK $worst_path]
puts "FINAL_WNS=$final_wns"
if {$final_wns < 0.0} {
    error "Timing constraints are not met; refusing to write GRU bitstream."
}

write_checkpoint -force [file join $output_dir "gru_final_impl.dcp"]
write_bitstream -force [file join $output_dir "GRU_Fall_Detect_MODERATE_FINAL.bit"]
puts "GRU_BITSTREAM_RESULT=OK"
exit
