#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>
#include <cfloat>

/*****************************************************************************
 *  Fused kernel: RTN quantise  → LUT remap  → 8×32-bit BVR packing
 *  Each block handles 1×128-elem group   (blockDim = 128 threads = 4 warps)
 *****************************************************************************/

#define GROUP_SIZE 128
#define GROUPS_PER_WARP 4
#define WARPS_PER_BLOCK 4
#define GROUPS_PER_BLOCK (GROUPS_PER_WARP * WARPS_PER_BLOCK) 
#define WARP_SZ    32
#define FULL_MASK  0xFFFFFFFFu

// __constant__ __device__ uint8_t IDX_LUT[64] = {
//       0,  1,  2,  3,  4,  5,  6,  7,
//       8,  9, 10, 11, 12, 13, 14, 15,
//      16, 17, 18, 19, 20, 21, 22, 23,
//      24, 25, 26, 27, 28, 29, 30, 31,
//       0,
//      16, 32, 48, 64, 80, 96,112,128,
//     144,160,176,192,208,224,240,  0,
//      16, 32, 48, 64, 80, 96,112,128,
//     144,160,176,192,208,224,240
// };

__device__ __forceinline__ float warpMax(float v)
{
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1)
        v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, ofs));
    return v;
}


__global__ void fused_rtn7_lut_bvr(const __half  *__restrict__ x,
                                  uint32_t      *__restrict__ out_bvr,
                                  float         *__restrict__ scales,
                                  int            num_groups)
{
    if (blockIdx.x >= num_groups) return;

    const int gid  = blockIdx.x;           // group id
    const int tid  = threadIdx.x;          // 0-127
    const int lane = tid & 31;             // 0-31
    const int warp = tid >> 5;             // 0-3
    const int base = gid * GROUP_SIZE;

    /* ── 1. group max-abs ────────────────────────────────────────────────── */
    float v    = __half2float(x[base + tid]);
    float amax = warpMax(fabsf(v));

    __shared__ float warp_max[4];
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();

    if (warp == 0) {
        // Only lanes 0-3 load valid data; others load a neutral element
        float gmax = (lane < 4) ? warp_max[lane] : -FLT_MAX;
        gmax       = warpMax(gmax);      // second-level reduction inside warp 0
        if (lane == 0) warp_max[0] = gmax;
    }
    __syncthreads();
    const float gmax = warp_max[0];
    const float s    = gmax / 63.f + 1e-10f;
    if (tid == 0) scales[gid] = s;

    /* ── 2. quantise + LUT ───────────────────────────────────────────────── */
     //int idx = __float2int_rn(v / s) + 31;          // 0-62
    // idx     = max(0, min(63, idx));                // clamp
    // uint8_t val = IDX_LUT[idx];
    uint8_t val = static_cast<uint8_t>(__float2int_rn(v / s) + 191); // [-63, 2^6, 2^5, .... 2^1, 2^0]. 191 = 128 + 63

    /* ── 3. bit-vector pack (8 ballots) ─────────────────────────────────── */
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        uint32_t bitvec = __ballot_sync(FULL_MASK, (val >> bit) & 1);
        if (lane == 0)
            out_bvr[gid * 32 + warp * 8 + bit] = bitvec;
    }
}

template <int nRTN = 7>
__device__ __forceinline__
void encode_rtn7_group_warp(const __half* __restrict x,   // 128 vals
                            uint32_t*       __restrict bvr,   // 4×(_nRTN) words
                            float*          __restrict scale) // 1 word
{
    constexpr int _nRTN = nRTN + 1;
    const int lane = threadIdx.y * blockDim.x + threadIdx.x;          // 0‥31

    /* 1) local abs-max (4 값) */
    float local_max = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v = __half2float(x[lane + i * 32]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    /* 2) warp-reduce max → lane 0 */
    #pragma unroll
    for (int ofs = 16; ofs; ofs >>= 1)
        local_max = fmaxf(local_max,
                          __shfl_down_sync(FULL_MASK, local_max, ofs));

    float grp_max = __shfl_sync(FULL_MASK, local_max, 0);   // 모든 lane에 전파
    float grp_scale = fmaxf(grp_max / 63.f, 1e-30f);
    if (lane == 0) *scale = grp_scale;

    /* 3) 4-chunk 양자화 & ballot */
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        float v = __half2float(x[lane + chunk * 32]);

        int q = __float2int_rn(v / grp_scale);
        q = max(-63, min(63, q));                 //  signed 7-bit
        uint8_t val = static_cast<uint8_t>(q + 191);  // 0x80|(q+63)

        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = __ballot_sync(FULL_MASK, (val >> bit) & 1);
            if (lane == 0) bvr[chunk * 8 + bit] = mask;
        }
    }
}


__global__ void fused_rtn7_lut_bvr_warp(const __half* __restrict x,
                                        uint32_t*    __restrict out_bvr,
                                        float*       __restrict scales,
                                        int num_groups)            // K/128
{
    const int warp_id = threadIdx.z;                     // 0‥3
    const int base_group = (blockIdx.x * GROUPS_PER_BLOCK) +
                           warp_id * GROUPS_PER_WARP;    // warp당 8개 시작점

    #pragma unroll
    for (int g = 0; g < GROUPS_PER_WARP; ++g) {
        int gid = base_group + g;
        if (gid >= num_groups) break;                    // 범위 체크

        const __half* x_ptr = x + gid * GROUP_SIZE;
        uint32_t* bvr_ptr   = out_bvr + gid * 4 * 8;     // 4×_nRTN (8)
        float*    scl_ptr   = scales   + gid;

        encode_rtn7_group_warp<7>(x_ptr, bvr_ptr, scl_ptr);
    }
}




void launch_fused_rtn_lut_bvr(
    const __half* x,
    uint32_t* out_bvr,
    float* scales,
    int num_groups = 16,
    int nRTN = 7)
{
    // dim3 grid(num_groups), block(GROUP_SIZE);

    // if (nRTN == 7){
    //     fused_rtn7_lut_bvr<<<grid, block>>>(x, out_bvr, scales, num_groups);
    // }
    // else{
    //     throw std::runtime_error("launch_fused_rtn_lut_bvr: nRTN value not implemented");
    // }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    //     throw std::runtime_error("CUDA error");
    // }

    constexpr int THREADS_X = 4;
    constexpr int THREADS_Y = 8;
    constexpr int THREADS_Z = 4;                // warp 4개
    dim3 block(THREADS_X, THREADS_Y, THREADS_Z);

    int num_blocks = (num_groups + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK;
    dim3 grid(num_blocks);

    fused_rtn7_lut_bvr_warp<<<grid, block>>>(x, out_bvr, scales, num_groups);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA launch error: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA launch failed");
    }
}