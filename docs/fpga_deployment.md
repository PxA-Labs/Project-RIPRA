# FPGA Deployment Guide for RIPRA

## Overview

FPGA acceleration targets sub-microsecond latency for the most compute-intensive pipeline stages: centroiding and matrix-vector multiply. A Xilinx Kintex-7 / Artix-7 class device (or Intel Cyclone V / Arria 10) provides sufficient logic for 127-spot centroiding + 20×254 matrix multiply at 10+ kHz frame rates.

## Pipeline Partitioning

```
Camera → [FPGA: centroid + matrix mul] → [CPU: modal reconstruction + DM map]
         ~~ < 2 µs ~~                    ~~ < 50 µs ~~
```

| Stage           | Latency Target | Implementation             |
|-----------------|----------------|----------------------------|
| Pixel readout   | 1 µs           | LVDS / MIPI Rx             |
| Centroid (127)  | 0.5 µs         | Pipelined TCoG + peak find |
| Matrix Mv Mul   | 0.3 µs         | Systolic array 20×254×32b  |
| Modal recon     | 10 µs          | ARM Cortex / CPU (soft/hard)|

## 1. Centroiding (TCoG + Peak Detection)

### Algorithm (VHDL)
- **Threshold + Center of Gravity** (TCoG) in a single pass
- Spot window: 11×11 pixels (from WFS geometry: lenslet pitch 300 µm → ~3 pix at 7.4 µm/pix)
- Each window processed in parallel (127 lanes)

```vhdl
-- Simplified TCoG datapath
entity centroid_engine is
    Port ( clk       : in  std_logic;
           pixel_in  : in  std_logic_vector(11 downto 0);  -- 12-bit ADC
           spot_id   : in  integer range 0 to 126;
           x_out     : out std_logic_vector(15 downto 0);  -- Q8.8 fixed
           y_out     : out std_logic_vector(15 downto 0);
           valid     : out std_logic);
end centroid_engine;

architecture pipelined of centroid_engine is
    type linebuf_t is array (0 to 10) of std_logic_vector(11 downto 0);
    signal linebuf : linebuf_t;
    signal accum_x, accum_y, accum_w : unsigned(31 downto 0);
begin
    process(clk) is
        variable sx, sy, sw : unsigned(31 downto 0);
    begin
        if rising_edge(clk) then
            -- Pipeline stage 1: load line buffer
            pixel_in <= linebuf(0);
            for i in 0 to 9 loop
                linebuf(i) <= linebuf(i+1);
            end loop;

            -- Pipeline stage 2: accumulate moments (threshold=ADC_mean*1.5)
            if pixel_in > THRESH then
                sx := sx + to_unsigned(col * pixel_in, 32);
                sy := sy + to_unsigned(row * pixel_in, 32);
                sw := sw + to_unsigned(pixel_in, 32);
            end if;

            -- Pipeline stage 3: divide (SW / SW, SX / SW, SY / SW)
            -- ... barrel divider, 11 cycles
        end if;
    end process;
end pipelined;
```

- **Latency**: 11+3 = 14 clock cycles @ 200 MHz → **70 ns per spot**
- **Total**: 127 spots × 70 ns = **8.9 µs** (sequential) or **70 ns** (fully parallel with 127 engines)
- **Resource**: ~500 LUTs + 2 DSP48 per engine → 127 × 500 = **63.5k LUTs** (fits Kintex-7)

### Parallel Implementation (Preferred)
- Instantiate 127 centroid engines, each with its own 11×11 window
- All engines run concurrently → **70 ns total** (= 1 window readout time)
- FPGA cost: moderate (63.5k LUTs ≈ 40% of XC7K325T)

## 2. Matrix-Vector Multiply (Z' × displacements)

### Kernel: `coeffs[20] = Zprime[20][254] × displacements[254]`

```vhdl
entity systolic_mv is
    Port ( clk     : in  std_logic;
           d_in    : in  std_logic_vector(31 downto 0);  -- float32
           c_out   : out std_logic_vector_vector(19 downto 0)(31 downto 0);
           done    : out std_logic);
end systolic_mv;

architecture systolic of systolic_mv is
    type matrix_t is array (0..19, 0..253) of std_logic_vector(31 downto 0);
    signal Z : matrix_t := (others => (others => (others => '0')));

    type pe_t is record
        acc : std_logic_vector(31 downto 0);
        d   : std_logic_vector(31 downto 0);
    end record;
    type pe_array is array (0..19) of pe_t;
    signal pes : pe_array;
begin
    -- Systolic array: 20 PEs, each computing sum(Z[row][k] * d[k])
    process(clk) is
        variable m : std_logic_vector(31 downto 0);
    begin
        if rising_edge(clk) then
            for row in 0 to 19 loop
                m := fp32_mul(Z(row, col), d_in);  -- DSP48
                pes(row).acc <= fp32_add(pes(row).acc, m);  -- accumulate
            end loop;
        end if;
    end process;
end systolic;
```

- **Latency**: 254 cycles + 16 pipeline = **270 cycles @ 200 MHz → 1.35 µs**
- **Resource**: 20 multipliers + 20 adders = **40 DSP48** (≈ 10% of XC7K325T)
- **Throughput**: 1 result per 254 clocks → 740k matrix-vec / sec @ 200 MHz

## 3. Modal Reconstruction & DM Map (CPU-side)

For flexibility, modal reconstruction and DM control remain on CPU:

| Step | CPU Time (est.) |
|------|----------------|
| Read FPGA centroid + matrix result (PCIe DMA) | 0.5 µs |
| r₀ / τ₀ estimation | 0.5 µs |
| DM command calculation | 0.3 µs |
| Send commands to DM DACs | 0.5 µs |
| **Total** | **1.8 µs** |

## 4. FPGA Resource Estimate (Xilinx XC7K325T)

| Block          | LUTs     | DSP48 | BRAM | Latency |
|----------------|----------|-------|------|---------|
| 127× centroid  | 63,500   | 254   | 0    | 70 ns   |
| Systolic MV    | 5,000    | 40    | 72   | 1.35 µs |
| Memory I/F     | 2,000    | 0     | 8    | —       |
| PCIe DMA       | 8,000    | 0     | 16   | 0.5 µs  |
| **Total**      | **78,500** (39%) | **294** (42%) | **96** (23%) | **~2 µs** |

XC7K325T has 203,800 LUTs, 840 DSP48, 445 BRAM — comfortable fit.

## 5. Build Flow

### Prerequisites
- Vivado 2024.1 (or later) with VHDL support
- Optional: Vitis HLS for C-to-RTL centroid acceleration

### Build
```bash
cd rippra/fpga
vivado -mode batch -source build_centroid.tcl
vivado -mode batch -source build_systolic.tcl
```

### Simulation
```bash
# Testbench for centroid engine
xsim tb_centroid -runall
# Expected: 127 (dx, dy) in 70 ns, accuracy < 0.01 pix RMS
```

### Integration
```tcl
# Top-level wrapper
add_files rippra/fpga/rtl/top.vhd
add_files rippra/fpga/rtl/centroid_engine.vhd
add_files rippra/fpga/rtl/systolic_mv.vhd
add_files rippra/fpga/constraints/timing.xdc
synth_design -top rippra_top -part xc7k325tffg900-2
place_design
route_design
write_bitstream -file rippra.bit
```

## 6. Host Software Interface

```c
// fpga_interface.h — FPGA control from C
typedef struct {
    void*  bar0;        // mmap'd PCIe BAR0
    size_t dma_buf;     // DMA buffer for frame data
} FPGACtx;

int fpga_init(FPGACtx* ctx);
int fpga_send_frame(FPGACtx* ctx, const uint16_t* pixels); // write to DDR
int fpga_wait_centroids(FPGACtx* ctx, float* dx, float* dy, int timeout_us);
int fpga_read_coeffs(FPGACtx* ctx, float coeffs[20]);

// Equivalent to software pipeline:
// centroid() + matrix_mul() in < 2 µs on FPGA
// Returns Zernike coefficients ready for DM mapping
```

## 7. Verification against Software Baseline

```bash
# Generate test vectors from software
python rippra/tools/generate_test_vectors.py --output fpga/tb/vectors.bin

# Run FPGA simulation
cd fpga/tb
vsim -c -do "run -all; compare centroids.txt expected_centroids.txt"
# RMS error should be < 0.01 pix (IEEE 754 float approx)

# Compare matrix-mul output
vsim -c -do "run -all; compare coeffs_out.txt expected_coeffs.txt"
# RMS error should be < 1e-6 rad
```

## 8. Performance Summary

| Metric | FPGA | CPU (GPU) |
|--------|------|-----------|
| Centroid latency | 70 ns (parallel) | 805 µs |
| Matrix Mv latency | 1.35 µs | 5 µs |
| Total pipeline | ~2 µs | ~900 µs |
| Frame rate | 500 kHz | 1.1 kHz |
| Power | ~5 W | ~30 W (CPU) |

## 9. Future: Direct Camera Link

For ultimate latency, the CMOS sensor can interface directly via LVDS:

```
Sensor LVDS → FPGA deserializer → centroid engine → PCIe DMA → CPU
              ~~ < 100 ns ~~
```

Eliminates the frame buffer readout bottleneck (currently ~300 µs for 648×492 via USB). With direct LVDS: **~5 µs** full-frame readout at 200 MHz pixel clock.
