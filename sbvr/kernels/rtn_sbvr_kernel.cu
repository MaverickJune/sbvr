#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>
#include <cfloat>

#define GROUP_SIZE 128

#define BLOCK_PER_SM 16
#define THREAD_PER_WARP 32
#define K_PER_BVR 4
#define _1xtN_tN 4
#define WARP_PER_BLOCK 4

#define FULL_MASK  0xFFFFFFFFu

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

/*****************************************************************************
 *  RTN-SBVR GEMM kernel
 *****************************************************************************/
__constant__ __half RTN_7_PIVOT[8] = {
    __float2half( 1.0f),  __float2half( 2.0f),  __float2half( 4.0f),
    __float2half( 8.0f),  __float2half(16.0f),  __float2half(32.0f),
    __float2half(64.0f),  __float2half(-63.0f)
};

template <typename RIndexT, int RNumSums, int TileN>
__global__ void rtn_7_sbvr_1xtN_mm_T(
    uint32_t *__restrict__ l_bvr, float *__restrict__ l_scales,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int N, int K)
{
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
        throw std::runtime_error("Not implemented nRTN values")
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
                launch_rtn_sbvr_1xtN_mm_T_kernel_wrapper<uint8_t, 4>(
                    l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, nRTN,
                    bias, out, N, K, device_id);
                break;
            default:
                throw std::runtime_error("Unsupported r_num_sums value");
        }
    }
}