set project_dir [file normalize [file join [file dirname [info script]] ".."]]
set bit_file [file join $project_dir "reports" "final_bitstream" "GRU_Fall_Detect_MODERATE_FINAL.bit"]

if {![file exists $bit_file]} {
    puts "GRU_PROGRAM_RESULT=BITSTREAM_NOT_FOUND"
    exit 2
}

open_hw_manager
connect_hw_server -allow_non_jtag
set targets [get_hw_targets]
if {[llength $targets] == 0} {
    puts "GRU_PROGRAM_RESULT=NO_TARGET"
    exit 3
}

current_hw_target [lindex $targets 0]
open_hw_target
set devices [get_hw_devices xc7a35t*]
if {[llength $devices] == 0} {
    puts "GRU_PROGRAM_RESULT=NO_XC7A35T_DEVICE"
    exit 4
}

current_hw_device [lindex $devices 0]
refresh_hw_device -update_hw_probes false [current_hw_device]
set_property PROGRAM.FILE $bit_file [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
puts "GRU_PROGRAM_RESULT=OK"
puts "GRU_PROGRAMMED_BITSTREAM=$bit_file"
exit
