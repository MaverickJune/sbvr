#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>
#include <cfloat>
#include <cmath>

// #include "rtn_constants.cuh"

#define GROUP_SIZE 128

#define BLOCK_PER_SM 16
#define THREAD_PER_WARP 32
#define K_PER_BVR 4
#define _1xtN_tN 4
#define WARP_PER_BLOCK 4
#define WARP_SZ 32

#define FULL_MASK  0xFFFFFFFFu

// variables for fused launching
constexpr int MAX_NUM_GROUPS = 256; //32768
constexpr int WORDS_PER_GRP = 4 * (7 + 1);

__device__ uint32_t d_bvr[MAX_NUM_GROUPS * WORDS_PER_GRP];
__device__ float    d_scales[MAX_NUM_GROUPS];

uint32_t* d_bvr_ptr;
float* d_scales_ptr;

__device__ __host__ __forceinline__ uint32_t div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ float warpMax(float v)
{
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1)
        v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, ofs));
    return v;
}

template <int NUM_SUMS>
struct bvrs;

template <>
struct bvrs<2> {
    int2 data;
    __device__ __forceinline__ int get(int idx) const 
    {
        return idx == 0 ? data.x : data.y;
    }
};
template <>
struct bvrs<4> {
    int4 data;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data.x;
        else if (idx == 1) return data.y;
        else if (idx == 2) return data.z;
        else return data.w;
    }
};
template <>
struct bvrs<6> {
    int2 data0;
    int2 data1;
    int2 data2;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data1.x;
        else if (idx == 3) return data1.y;
        else if (idx == 4) return data2.x;
        else return data2.y;
    }
};
template <>
struct bvrs<8> {
    int4 data0;
    int4 data1;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data0.z;
        else if (idx == 3) return data0.w;
        else if (idx == 4) return data1.x;
        else if (idx == 5) return data1.y;
        else if (idx == 6) return data1.z;
        else return data1.w;
    }
};
template <>
struct bvrs<10> {
    int2 data0;
    int2 data1;
    int2 data2;
    int2 data3;
    int2 data4;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data1.x;
        else if (idx == 3) return data1.y;
        else if (idx == 4) return data2.x;
        else if (idx == 5) return data2.y;
        else if (idx == 6) return data3.x;
        else if (idx == 7) return data3.y;
        else if (idx == 8) return data4.x;
        else return data4.y;
    }
};

template <int NUM_SUMS>
struct coeffs {
    __half2 coeff[NUM_SUMS / 2];
};

extern int device_count;
extern cudaDeviceProp cuda_prop_list[16];

/* helper so the table stays readable */
static constexpr __half h(u_int16_t bits) noexcept
{
    return __half{ __half_raw{ bits } };
}

__device__ __constant__ __half RTN_7_PIVOT[8] = {
    h(0x3C00),   //  +1.0
    h(0x4000),   //  +2.0
    h(0x4400),   //  +4.0
    h(0x4800),   //  +8.0
    h(0x4C00),   // +16.0
    h(0x5000),   // +32.0
    h(0x5400),   // +64.0
    h(0xD800)    // −128.0
    // h(0xD3E0)    // -63.0  h(0xD800)    // −128.0
};

/*****************************************************************************
 *  RTN-SBVR GEMM kernel
 *****************************************************************************/

template <typename RIndexT, int RNumSums, int TileN>
__global__ void rtn_7_sbvr_1xtN_mm_T(
    uint32_t *__restrict__ l_bvr, float *__restrict__ l_scales,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int N, int K)
{
    // /* --------------------------------- DEBUG PRINT */
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0)        // only one thread prints
    // {
    //     printf("---- RTN_7_PIVOT ----\n");
    //     #pragma unroll
    //     for (int i = 0; i < 8; ++i)
    //         printf("  [%d] = %f\n", i, __half2float(RTN_7_PIVOT[i]));
    //     printf("---------------------\n");
    // }
    // __syncthreads();            // optional, keeps all threads aligned here
    // return;                     // early-out so kernel does *nothing* else

    constexpr int LNumSums = 8;

    const int tblock_per_N = (N + TileN - 1) / TileN;
    const int bvr_per_K = K / K_PER_BVR;

    for (int tblock_id = blockIdx.x * blockDim.z + threadIdx.z; 
        tblock_id < tblock_per_N;
         tblock_id += gridDim.x * blockDim.z)
    {
        const int n = tblock_id * TileN + threadIdx.x;
        float sum = 0.0f;
        if (n < N)
        {
            for (int bvr_idx = threadIdx.y; bvr_idx < bvr_per_K; 
                    bvr_idx += blockDim.y)
            {
                // If all bits of l_bvr and r_bvr are set, this may overflow.
                uchar4 popc_cache [LNumSums / 2][RNumSums / 2] = {};
                #pragma unroll
                for (int k = 0; k < K_PER_BVR; k++)
                {
                    const int k_idx = bvr_idx * K_PER_BVR + k;
                    const bvrs<LNumSums> l_bvrs = 
                        *(bvrs<LNumSums>*)(&l_bvr[k_idx * LNumSums]);
                    const bvrs<RNumSums> r_bvrs = 
                        *(bvrs<RNumSums>*)(&r_bvr[(k_idx * N + n) * RNumSums]);
                    #pragma unroll
                    for (int l_idx = 0; l_idx < LNumSums / 2; l_idx++)
                    {
                        #pragma unroll
                        for (int r_idx = 0; r_idx < RNumSums / 2; r_idx++)
                        {
                            const uint32_t l_0 = l_bvrs.get(l_idx * 2);
                            const uint32_t l_1 = l_bvrs.get(l_idx * 2 + 1);
                            const uint32_t r_0 = r_bvrs.get(r_idx * 2);
                            const uint32_t r_1 = r_bvrs.get(r_idx * 2 + 1);
                            popc_cache[l_idx][r_idx].x += __popc(l_0 & r_0);
                            popc_cache[l_idx][r_idx].y += __popc(l_1 & r_1);
                            popc_cache[l_idx][r_idx].z += __popc(l_0 & r_1);
                            popc_cache[l_idx][r_idx].w += __popc(l_1 & r_0);
                        }
                    }
                }

                // get the coeffs and scale
                const coeffs<LNumSums> l_coeffs = 
                    *(coeffs<LNumSums>*)(RTN_7_PIVOT);
                const float l_scale = __ldg(&l_scales[bvr_idx]);
                const int r_coeff_i = __ldg(&r_coeff_idx[bvr_idx * N + n]);
                const coeffs<RNumSums> r_coeffs = 
                    *(coeffs<RNumSums>*)(&r_coeff_cache[r_coeff_i * RNumSums]);

                #pragma unroll
                for (int l_idx = 0; l_idx < LNumSums / 2; l_idx++)
                {
                    #pragma unroll
                    for (int r_idx = 0; r_idx < RNumSums / 2; r_idx++)
                    {
                        const __half2 popc_h_0 = 
                            __halves2half2(
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].x),
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].y));
                        const __half2 popc_h_1 = 
                            __halves2half2(
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].z),
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].w));
                        const __half2 l_coeff = __hmul2(l_coeffs.coeff[l_idx],
                                                        __float2half2_rn(l_scale));
                        const __half2 r_coeff = r_coeffs.coeff[r_idx];
                        const __half2 coeff_0 = __hmul2(l_coeff, r_coeff);
                        const __half2 coeff_1 =
                            __hmul2(l_coeff, 
                                        __halves2half2(__high2half(r_coeff), 
                                                       __low2half(r_coeff)));
                        const __half2 mult_sum = __hfma2(coeff_0, popc_h_0, 
                                                    __hmul2(coeff_1, popc_h_1));                               
                        sum += __half2float(mult_sum.x) + 
                               __half2float(mult_sum.y);
                    }
                }
            }
        }
        #pragma unroll
        for (int i = (THREAD_PER_WARP / TileN) / 2; i > 0; i /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, i * TileN);

        if (threadIdx.y == 0 && n < N)
        {
            __half bias_val = (bias != nullptr ? bias[n] : __float2half(0.0f));
            bias_val = __hadd(__float2half(sum), bias_val);  
            out[n] = bias_val; 
        }
    }
}

/*****************************************************************************
 *  Fused kernel: RTN quantise  → LUT remap  → 8×32-bit BVR packing
 *  Each block handles 1×128-elem group   (blockDim = 128 threads = 4 warps)
 *****************************************************************************/
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
    const float s    = gmax / 127.f + 1e-10f;
    // const float s    = gmax / 63.f + 1e-10f;
    if (tid == 0) scales[gid] = s;

    /* ── 2. quantise + LUT ───────────────────────────────────────────────── */
    // uint8_t val = static_cast<uint8_t>(__float2int_rn(v / s) + 191);
    uint8_t val = static_cast<uint8_t>(__float2int_rn(v / s) + 128);

    /* ── 3. bit-vector pack (8 ballots) ─────────────────────────────────── */
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        uint32_t bitvec = __ballot_sync(FULL_MASK, (val >> bit) & 1);
        if (bit == 7)
            bitvec = ~bitvec;
        if (lane == 0)
            out_bvr[gid * 32 + warp * 8 + bit] = bitvec;
    }
}

template <typename RIndexT, int RNumSums>
void launch_rtn_sbvr_1xtN_mm_T_kernel_wrapper(
    uint32_t* l_bvr, float* l_scales,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    int nRTN,
    __half* bias, __half* out,
    int N, int K,
    int device_id = 0)
{
    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads = {_1xtN_tN, THREAD_PER_WARP / _1xtN_tN, WARP_PER_BLOCK};

    if (nRTN == 7) {
        rtn_7_sbvr_1xtN_mm_T<RIndexT, RNumSums, _1xtN_tN><<<blocks, threads>>>(
            l_bvr, l_scales,
            r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
            bias, out,
            N, K);
    }
    else {
        throw std::runtime_error("Not implemented nRTN values");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void launch_rtn_sbvr_1xtN_mm_T(
    uint32_t* l_bvr, float* l_scales,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    int r_num_sums,
    int r_cache_size,
    int nRTN,
    __half* bias, __half* out,
    int N, int K,
    int device_id = 0)
{
    const bool use_r_uint8 = (r_cache_size <= 256);
    
    if (use_r_uint8) {
        switch (r_num_sums) {
            case 4:
                launch_rtn_sbvr_1xtN_mm_T_kernel_wrapper<uint8_t, 4>(
                    l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, nRTN,
                    bias, out, N, K, device_id);
                break;
            default:
                throw std::runtime_error("Unsupported r_num_sums value");
        }
    } 
    else {
       switch (r_num_sums) {
            case 4:
                launch_rtn_sbvr_1xtN_mm_T_kernel_wrapper<uint16_t, 4>(
                    l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, nRTN,
                    bias, out, N, K, device_id);
                break;
            default:
                throw std::runtime_error("Unsupported r_num_sums value");
        }
    }
}

void launch_fused_rtn_lut_bvr(
    const __half* x,
    uint32_t* out_bvr,
    float* scales,
    int num_groups = 16,
    int nRTN = 7)
{
    dim3 grid(num_groups), block(GROUP_SIZE);

    if (nRTN == 7){
        fused_rtn7_lut_bvr<<<grid, block>>>(x, out_bvr, scales, num_groups);
    }
    else{
        throw std::runtime_error("launch_fused_rtn_lut_bvr: nRTN value not implemented");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

/*****************************************************************************
 * launch the fused kernel
 *****************************************************************************/
template <typename RIndexT, int RNumSums>
void launch_fused_rtn_7_sbvr_1xtN_mm_T_kernel_wrapper(
    __half* x,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int N, int K,
    int device_id = 0)
{
    // grid and threadblock config for mm_T
    int blocks_mm_T = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads_mm_T = {_1xtN_tN, THREAD_PER_WARP / _1xtN_tN, WARP_PER_BLOCK};

    // set num_groups
    int num_groups = K / GROUP_SIZE;

    // grid and threadblock config for rtn_input
    dim3 grid_rtn_input(num_groups), block_rtn_input(GROUP_SIZE);

    // launch the kernels in sequential manner
    fused_rtn7_lut_bvr<<<grid_rtn_input, block_rtn_input>>>(x, d_bvr_ptr, d_scales_ptr, num_groups);
    rtn_7_sbvr_1xtN_mm_T<RIndexT, RNumSums, _1xtN_tN><<<blocks_mm_T, threads_mm_T>>>(
                d_bvr_ptr, d_scales_ptr,
                r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
                bias, out,
                N, K / 32);

}

void launch_fused_rtn_sbvr_1xtN_mm_T(
    __half* x,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    int r_num_sums,
    int r_cache_size,
    int nRTN,
    __half* bias, __half* out,
    int N, int K,
    int device_id = 0)
{

    const bool use_r_uint8 = (r_cache_size <= 256);

    if (use_r_uint8 && nRTN == 7) {
        if (r_num_sums == 4) {
            launch_fused_rtn_7_sbvr_1xtN_mm_T_kernel_wrapper<uint8_t, 4>(
                x, r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out, N, K, device_id);
        }
        else {
            throw std::runtime_error("Unsupported r_num_sums value");
        }
    } 
    else if(use_r_uint8 == false && nRTN == 7) {
        if (r_num_sums == 4) {
            launch_fused_rtn_7_sbvr_1xtN_mm_T_kernel_wrapper<uint16_t, 4>(
                x, r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out, N, K, device_id);
        }
        else {
            throw std::runtime_error("Unsupported r_num_sums value");
        }
    } 
    else {
        throw std::runtime_error("Unsupported nRTN value");
    }
}


// for fused launching
void launch_cudaGetSymbolAddress_wrapper()
{
    cudaGetSymbolAddress((void**)&d_bvr_ptr, d_bvr);
    cudaGetSymbolAddress((void**)&d_scales_ptr, d_scales);
}