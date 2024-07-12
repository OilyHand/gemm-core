# Constants for zedboard spec in vta
## Accumulation 관련

|명칭|값|
|-|-|
|||
|||

## Weight 관련

## Input 관련

## Output 관련

## matrix 관련


```
VTA_LOG_INP_WIDTH=3
VTA_INP_WIDTH=8

VTA_LOG_WGT_WIDTH=3
VTA_WGT_WIDTH=8

VTA_LOG_ACC_WIDTH=5
VTA_ACC_WIDTH=32

VTA_LOG_OUT_WIDTH=3
VTA_OUT_WIDTH=8

VTA_LOG_BATCH=0
VTA_BATCH=1

VTA_LOG_BLOCK_IN=4
VTA_BLOCK_IN=16
VTA_LOG_BLOCK_OUT=4
VTA_BLOCK_OUT=16

input tile  : 8  bits
weight tile : 8  bits
accum tile  : 32 bits
output tile : 8  bits

input tensor  : 1 * 16  = 16  tiles 
weight tensor : 16 * 16 = 256 tiles
accum tensor  : 1 * 16  = 16  tiles
output tensor : 1 * 16  = 16  tiles


VTA_INP_MATRIX_WIDTH = (VTA_INP_WIDTH * VTA_BATCH * VTA_BLOCK_IN)
                     = 8 * 1 * 16 = 128 bits
VTA_WGT_MATRIX_WIDTH = (VTA_WGT_WIDTH * VTA_BLOCK_OUT * VTA_BLOCK_IN)
                     = 8 * 16 * 16 = 2048 bits 
VTA_ACC_MATRIX_WIDTH = (VTA_ACC_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
                     = 32 * 1 * 16 = 512 bits
VTA_OUT_MATRIX_WIDTH = (VTA_OUT_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
                     = 8 * 1 * 16 = 128 bits


INP_MAT_AXI_RATIO = (VTA_INP_MATRIX_WIDTH / VTA_BUS_WIDTH)
                  = 128 / 64 = 2
WGT_MAT_AXI_RATIO = (VTA_WGT_MATRIX_WIDTH / VTA_BUS_WIDTH)
                  = 2048 / 64 = 32
ACC_MAT_AXI_RATIO = (VTA_ACC_MATRIX_WIDTH / VTA_BUS_WIDTH)
                  = 512 / 64 = 8
OUT_MAT_AXI_RATIO = (VTA_OUT_MATRIX_WIDTH / VTA_BUS_WIDTH)
                  = 128 / 64 = 2

VTA_UOP_BUFF_DEPTH = (VTA_UOP_BUFF_SIZE / VTA_UOP_ELEM_BYTES)
                  = 2^15 / 4 = 2^13 = 8192 = 2^5 8 2^8 = 32 * 256
VTA_WGT_BUFF_DEPTH = (VTA_WGT_BUFF_SIZE / VTA_WGT_ELEM_BYTES)
                  = 2^18 / (2^11 / 2^3) = 1024
VTA_INP_BUFF_DEPTH = (VTA_INP_BUFF_SIZE / VTA_INP_ELEM_BYTES)
                  = 2^15 / (2^7 / 2^3) = 2048
VTA_ACC_BUFF_DEPTH = (VTA_ACC_BUFF_SIZE / VTA_ACC_ELEM_BYTES)
                  = 2^17 / (2^9/2^3) = 2048

VTA_UOP_WIDTH = 2^5 = 32
VTA_UOP_ELEM_BYTES = 32 / 8 = 4


VTA_LOG_INS_WIDTH : 7
VTA_INS_WIDTH : 2^7 = 128 --> uint128_t

VTA_LOG_UOP_WIDTH : 5
VTA_UOP_WIDTH : 2^5 = 32

VTA_LOG_BUS_WIDTH : 6
VTA_BUS_WIDTH : 2^6 = 64

VTA_LOG_ACC_BUFF_DEPTH
 = VTA_LOG_ACC_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_OUT - VTA_LOG_ACC_WIDTH + 3
 = 17 - 0 - 4 - 5 + 3 = 11

VTA_LOG_INP_BUFF_DEPTH
 = VTA_LOG_INP_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_IN - VTA_LOG_INP_WIDTH + 3
 = 15 - 0 - 4 - 3 + 3 = 11

VTA_LOG_WGT_BUFF_DEPTH
 = VTA_LOG_WGT_BUFF_SIZE - VTA_LOG_BLOCK_OUT - VTA_LOG_BLOCK_IN - VTA_LOG_WGT_WIDTH + 3
 = 18 - 4 - 4 - 3 + 3 = 10

micro-op field
|accumulator index|input index|weight index|
|11               |11         |10          |
|0              10|11       21|22        31|


VTAGemInsn := unsigned <128> bit
acc_idx_T  := unsigned <12>  bit
inp_idx_T  := unsinged <12>  bit
wgt_idx_T  := unsinged <11>  bit
uop_T      := unsinged <32>  bit
acc_T signed 32 bit
sum_T = VTA_WGT_WIDTH+VTA_INP_WIDTH+VTA_LOG_BLOCK_IN+1 = 8+8+4+1 = 21 bit signed
mul_T = VTA_WGT_WIDTH+VTA_INP_WIDTH+1 = 8 + 8 + 1(for sign bit) = 17 bit signed
out_T = signed 8 bit
```

```cpp
// instruction 하나 들어왔을 때의 동작
  // micro-op cache, register file, input buffer, weight buffer에 fetch 되어야 함
void gemm (
  uint128_t insn_raw,             // gemm instruction
  uint32_t  uop_mem[2^13],        // micro-op cache  (AXI)
  uint64_t  acc_mem[2^11][2^3],   // register file   (AXI)
  uint64_t  inp_mem[2^11][2^1],   // input buffer    (AXI)
  uint64_t  wgt_mem[2^10][2^6],   // weight buffer   (AXI)
  uint64_t  out_mem[2^11][2^1])   // output buffer   (AXI)

  uint128_t insn = insn_raw;
  /*
  insn
    [2:0]     opcode
    [3]       pop_prev_dep
    [4]       pop_next_dep
    [5]       push_prev_dep
    [6]       push_next_dep
    [7]       reset_reg
    [20:8]    uop_bgn
    [34:21]   uop_end
    [48:35]   iter_out
    [62:49]   iter_in
    [73:63]   dst_factor_out
    [84:74]   dst_factor_in
    [95:85]   src_factor_out
    [106:96]  src_factor_in
    [116:107] wgt_factor_out
    [126:117] wgt_factor_in
    [127]     *unused
  */

  uint12_t dst_offset_out // init = 0
  uint12_t src_offset_out // init = 0
  uint11_t wgt_offset_out // init = 0

  // loop `insn.iter_out` times
    uint12_t dst_offset_in // init for loop = dst_offset_out;
    uint12_t src_offset_in // init for loop = src_offset_out;
    uint11_t wgt_offset_in // init for loop = wgt_offset_out;

    // loop `insn.iter_in` times
      // micro-op process upc : insn.uop_bgn ~ insn.uop_end 
          // fetch to micro-op cache
      // pipeline
      uint32_t uop = uop_mem[upc];

  for (int b = 0; b < VTA_BATCH; b++) {
          for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) {
            // Initialize the accumulator values
            acc_T accum = a_tensor[b][oc];
            // Dot product sum
            sum_T tmp = 0;
            // Inner matrix multiplication loop (input channel/feature)
            for (int ic = 0; ic < VTA_BLOCK_IN; ic++) {
              wgt_T w_elem = w_tensor[oc][ic];
              inp_T i_elem = i_tensor[b][ic];
              mul_T prod_dsp = i_elem * w_elem;
              tmp += (sum_T) prod_dsp;
            }

        // Write the results back into accumulator
        write_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, a_tensor, acc_mem);
        // Write the results back in the output buffer
        write_tensor<bus_T, out_T, acc_idx_T, VTA_BUS_WIDTH, VTA_OUT_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, o_tensor, out_mem);
      }
      // Update offsets
      dst_offset_in += insn.dst_factor_in;
      src_offset_in += insn.src_factor_in;
      wgt_offset_in += insn.wgt_factor_in;
    }
    // Update offsets
    dst_offset_out += insn.dst_factor_out;
    src_offset_out += insn.src_factor_out;
    wgt_offset_out += insn.wgt_factor_out;
  }
}
```
