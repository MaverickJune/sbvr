#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>
#include <cfloat>

// #include "rtn_constants.cuh"

#define GROUP_SIZE 128

#define BLOCK_PER_SM 16
#define THREAD_PER_WARP 32
#define K_PER_BVR 4
#define _1xtN_tN 4
#define WARPS_PER_BLOCK 4

#define FULL_MASK  0xFFFFFFFFu


#define GROUPS_PER_WARP 2
#define GROUPS_PER_BLOCK (GROUPS_PER_WARP * WARPS_PER_BLOCK) 

__device__ __forceinline__ float warpMax(float v)
{
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1)
        v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, ofs));
    return v;
}

__device__ __host__ __forceinline__ uint32_t div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
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
    h(0xD3E0)    // −63.0
};


template <int nRTN = 7>
__device__ __forceinline__
float encode_rtn7_group_warp(const __half* __restrict x,   // 128 vals
                             uint32_t*        __restrict bvr) // 4 × _nRTN words
{
    constexpr int _nRTN = nRTN + 1;          // nRTN = 7 → _nRTN = 8
    const int lane = threadIdx.y * blockDim.x + threadIdx.x; // (4,8,4) → 0‥31

    /* 1) local abs-max (4 값) */
    float local_max = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v = __half2float(x[lane + i * 32]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    /* 2) warp-reduce max → 모든 lane */
    #pragma unroll
    for (int ofs = 16; ofs; ofs >>= 1)
        local_max = fmaxf(local_max,
                          __shfl_down_sync(FULL_MASK, local_max, ofs));

    const float grp_max   = __shfl_sync(FULL_MASK, local_max, 0);
    const float grp_scale = fmaxf(grp_max / 63.f, 1e-30f);

    /* 3) 4-chunk 양자화 & ballot */
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        float    v   = __half2float(x[lane + chunk * 32]);
        int      q   = max(-63, min(63, __float2int_rn(v / grp_scale)));
        uint8_t  val = static_cast<uint8_t>(q + 191);         // 0x80 | (q+63)

        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = __ballot_sync(FULL_MASK, (val >> bit) & 1);
            if (lane == 0) bvr[chunk * 8 + bit] = mask; 
        }
    }

    return grp_scale;  
}


/*****************************************************************************
 * Fused RTN 7 + SBVR 1xT*N matrix multiplication kernel
 *****************************************************************************/

template <typename RIndexT, int RNumSums, int TileN, int _nRTN = 7>
__global__ void fused_rtn_7_sbvr_1xtN_mm_T(
        const __half* __restrict x,          // 1 × K
        const uint32_t* __restrict r_bvr,
        const RIndexT*  __restrict r_coeff_idx,
        const __half*   __restrict r_coeff_cache,
        const __half*   __restrict bias,       // (N) or nullptr
        __half*         __restrict out,        // (1×N)
        int N, int K)
{
    const int warp_id     = threadIdx.z;               // 0‥3
    const int num_groups  = K / GROUP_SIZE;            // K / 128

    // Shared memory for l_bvr and scales
    extern __shared__ uint32_t smem[];
    uint32_t* l_bvr = smem;
    float* l_scales = (float*)(l_bvr + num_groups * 4 * (_nRTN + 1));


    for (int gb = 0; gb < num_groups; gb += GROUPS_PER_BLOCK) {

        #pragma unroll
        for (int g = 0; g < GROUPS_PER_WARP; ++g) {

            int gid = gb + warp_id * GROUPS_PER_WARP + g;
            if (gid >= num_groups) break;

            const __half* x_ptr   = x + gid * GROUP_SIZE;

            uint32_t* bvr_ptr = l_bvr + gid  * 4 * (_nRTN+1);   // 4 × 8 words

            float* scl_gptr  = l_scales + gid; // 1 float

            float grp_scale = encode_rtn7_group_warp<_nRTN>(x_ptr, bvr_ptr);  // scale return
            if (threadIdx.x == 0 && threadIdx.y == 0)
                *scl_gptr = grp_scale;                      // global write

        }

        __syncthreads(); 
    }


    ////////////////////////////

    constexpr int LNumSums = 8;

    K = K / 32;

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
                // const float l_scale = __ldg(&l_scales[bvr_idx]);
                const float l_scale = l_scales[bvr_idx];
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




template <typename RIndexT, int RNumSums>
void launch_fused_rtn_sbvr_1xtN_mm_T_kernel_wrapper(
    __half* x,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int N, int K,
    int device_id = 0)
{
    int  num_groups   = K / 128; // K / GROUP_SIZE

    size_t shm_bytes =
      num_groups * 4 * (7+1) * sizeof(uint32_t)   // 4KB
    + num_groups              * sizeof(float);    // 128B

    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads = {_1xtN_tN, THREAD_PER_WARP / _1xtN_tN, WARPS_PER_BLOCK};


    fused_rtn_7_sbvr_1xtN_mm_T<RIndexT, RNumSums, _1xtN_tN, 7>
        <<<blocks, threads, shm_bytes>>>(
            x, r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
            bias, out, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in fused_rtn_sbvr_1xtN_mm_T_kernel_wrapper: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }
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

    const bool use_r_uint8 = (r_cache_size < 256);

    if(use_r_uint8) {
        if (r_num_sums == 8 && nRTN == 7) {
            launch_fused_rtn_sbvr_1xtN_mm_T_kernel_wrapper<uint8_t, 8>(
                x, r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out, N, K, device_id);
        } else {
            throw std::runtime_error("Unsupported r_num_sums value for uint8_t");
        }
    } else {
        if (r_num_sums == 8 && nRTN == 7) {
            launch_fused_rtn_sbvr_1xtN_mm_T_kernel_wrapper<uint16_t, 8>(
                x, r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out, N, K, device_id);
        } else {
            throw std::runtime_error("Unsupported r_num_sums value for uint16_t");
        }
    }

}