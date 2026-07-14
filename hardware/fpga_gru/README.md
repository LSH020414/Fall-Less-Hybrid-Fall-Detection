# Basys3 fixed-point GRU

This directory contains the synthesizable GRU inference hardware.

- `src/`: RTL
- `sim/`: Vivado simulation testbenches
- `mem/`: quantized weights and activation LUTs
- `model/`: source GRU checkpoint and exported weights
- `tools/`: quantization and packet utilities
- `constraints/`: Basys3 pins and 100MHz clock
- `scripts/`: project creation, build, and programming
- `bitstream/`: final verified `.bit`
- `reports/`: timing, utilization, and DRC reports

See [HARDWARE_GRU.md](../../docs/HARDWARE_GRU.md).

## Deployment profiles

- `GRU_Fall_Detect_TUNED115_FINAL.bit`: dataset-optimized profile
- `GRU_Fall_Detect_CONSERVATIVE_FINAL.bit`: live-demo conservative profile
- `GRU_Fall_Detect_MODERATE_FINAL.bit`: current balanced live profile

The moderate profile is currently programmed on the Basys3. It requires two
high-probability windows, so a single unstable pose frame is rejected while
real falls are detected more readily than with the conservative profile.
