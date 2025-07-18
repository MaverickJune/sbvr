#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <iostream>
#include <cstdint>
#include <cfloat>

using namespace nvcuda;

/* --- Default Necessary Configs --------------------------------------------------- */
#define GROUP_SIZE 128

#define BLOCK_PER_SM 16
#define THREAD_PER_WARP 32
#define K_PER_BVR 4 // BVR size 128, BVR size = K_PER_BVR * 32 
#define _1xtN_tN 4
#define WARP_PER_BLOCK 4
#define WARP_SZ 32

#define FULL_MASK  0xFFFFFFFFu

/* --- CTA configuration --------------------------------------------------- */
#define CTA_M 128
#define CTA_N 128
#define CTA_K 32
#define WARPS_PER_CTA 16
#define THREADS_PER_CTA 512   /* 16 warps */

/* --- Tiny helpers -------------------------------------------------------- */
__device__ __forceinline__ half  hzero() { return __float2half(0.f);}
__device__ __forceinline__ float fzero() { return 0.f;}

/* --- BVR and Coeff Structs -------------------------------------------------------- */
template <int NUM_SUMS>
struct bvrs;

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

template <int NUM_SUMS>
struct unpacked_coeffs {
    half coeff[NUM_SUMS];
};

/********************************************************************************************
 * Shared-memory panel loaders (edge-aware)
 ********************************************************************************************/
__device__ void load_panel_A_into_smem(
        half *As, const half *A,
        int cta_m, int k0,
        int M, int K)
{
    /* Each thread loads one element (128 × 32 = 4096 halfs) */
    const int tid = threadIdx.x;          // 0-512
    const int smem_col = tid & (CTA_K - 1);         // 0-31  (CTA_K)

    const int g_col = k0 + smem_col;     // may exceed K
    int g_row = -1;
    const int row_offset = tid >> 5;
    const int row_stride = blockDim.x >> 5;

    #pragma unroll
    for (int smem_row = 0; smem_row < CTA_M; smem_row += row_stride)
    {
        g_row = cta_m + smem_row + row_offset;
        half val = (g_row < M && g_col < K)
             ? A[g_row * K + g_col]
             : hzero();
        As[(smem_row + row_offset) * CTA_K + smem_col] = val;
    }
}

// currently, only works when RNumSums == 4, due to bvrs struct
template <typename RIndexT, int RNumSums>
__device__ void load_sbvrized_panel_B_into_smem(
    half *Bs,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, half* __restrict__ r_coeff_cache,
    int k0, int cta_n,
    int K, int N)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int smem_row = tid & (CTA_K - 1);

    const int k0_eff = k0 >> 5; //  divide 32
    const int bvr_idx = k0_eff >> 2; // divide 128

    int g_col = -1;
    const int col_offset = tid >> 5; // 0, 1, ... 15
    const int col_stride = blockDim.x >> 5; // 16 (512 / 32)
    
    #pragma unroll
    for (int smem_col = 0; smem_col < CTA_N; smem_col += col_stride)
    {
        // 1. get the column index, and define dequant_elem
        g_col = cta_n + smem_col + col_offset;
        half dequant_elem = hzero();

        if (g_col < N) {
            // 2. fetch the bvr values
            const bvrs<RNumSums> r_bvrs = 
                *(bvrs<RNumSums>*)(&r_bvr[(k0_eff * N + g_col) * RNumSums]);
            const int r_coeff_i = __ldg(&r_coeff_idx[bvr_idx * N + g_col]);
            const unpacked_coeffs<RNumSums> r_coeffs = 
                *(unpacked_coeffs<RNumSums>*)(&r_coeff_cache[r_coeff_i * RNumSums]);

            // 3. restore the designated element
            for (int r = 0; r < RNumSums; r++)
                dequant_elem = __hfma(__int2half_rn((r_bvrs.get(r) >> lane) & 1u), r_coeffs.coeff[r], dequant_elem);
        }

        // 4. write it down to smem, write it in column-major format
        Bs[(smem_col + col_offset) * CTA_K + smem_row] = dequant_elem;
    }
}

/* =========================================================================
 *  Guarded store of a 16 × 16 WMMA accumulator fragment
 * ========================================================================= */
__device__ void store_frag_as_fp16_guarded(
    const wmma::fragment<wmma::accumulator,16,16,16,float>& frag,
    half* __restrict__ C, int base_row, int base_col,
    int M, int N,
    half* __restrict smem)
{
    constexpr int FRAG_ELEMS = 16 * 16; // 256
    constexpr int FRAG_SIZE = FRAG_ELEMS;

    float* tmp_base = reinterpret_cast<float*>(           // <-- 4-B aligned
        smem + CTA_M * CTA_K          // skip As   (half)
             + CTA_K * CTA_N);        // skip Bs   (half)

    int warp_id = threadIdx.x >> 5;                   // 0-31
    float* tmp = tmp_base + warp_id * FRAG_SIZE;     // 1 KiB per warp

    wmma::store_matrix_sync(tmp, frag, 16, wmma::mem_row_major);
    __syncthreads();   // let the whole warp finish the store

    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int idx = lane; idx < FRAG_ELEMS; idx += 32) {  // 8 iters per lane
        int i = idx >> 4;      // idx / 16
        int j = idx & 15;      // idx % 16
        int row = base_row + i;
        int col = base_col + j;

        if (row < M && col < N) {
            C[row * N + col] = __float2half(tmp[idx]);
        }
    }
    __syncthreads();   // optional: keep lifetime tidy before next use

}

/********************************************************************************************
 * SBVR Prefill Kernel
 * SBVR does not target fast prefill
 * However, for completeness this kernel is necessary
 * This kernel does not change input into sbvr format, it decoes the weight and then compute
 * 
 * r_bvr: (K/32, N, num_sums)
 * r_coeff_idx: (K/group_size, N)
 * r_coeff_cache: (#coeffs, num_sums)
 * 
 * block: 512 threads inside, (512, 1, 1)
 * grid: [X, Y]
 * padded launch is needed to deal with leftovers
 ********************************************************************************************/

 template <typename RIndexT, int RNumSums>
 __global__ void sbvr_prefill_mm_T(
    __nv_bfloat16* __restrict__ x,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __nv_bfloat16* __restrict__ r_coeff_cache,
    __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int M, int N, int K)
{
    // 1. CTA indices and other important indices
    const int cta_m = blockIdx.y * CTA_M;
    const int cta_n = blockIdx.x * CTA_N;
    
    // 2. declare shared memory
    extern __shared__ half smem[];
    half *As = smem;                       /* 128 × 32 = 4096 halfs (M, K)  */
    half *Bs = smem + CTA_M * CTA_K;       /* 128 × 32 = 4096 halfs, col_major (N, K) !!  */

    // 3. Per-warp WMMA fragments
    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b[2];
    wmma::fragment<wmma::accumulator, 16,16,16, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(acc[i], fzero());

    // 4. loop over the K dimension in 32-wide slabs
    for (int k0 = 0; k0 < K; k0 += CTA_K)
    {
        /* 4-a. Cooperative load A and B panels into shared memory (edge-aware) */
        load_panel_A_into_smem(As, x, cta_m, k0, M, K);
        load_sbvrized_panel_B_into_smem<RIndexT, RNumSums>(Bs, r_bvr, r_coeff_idx, r_coeff_cache, k0, cta_n, K, N);
        __syncthreads();

        /* 4-b. Warp-level WMMA multiply-accumulate inside this slab */
        int warp_id = threadIdx.x >> 5;        /* 0-15                         */
        int lane_id = threadIdx.x & 31;       /* 0-31 within the warp        */
        int warp_row = warp_id & 3;          // 0‥3 → which 32-row band
        int warp_col = warp_id >> 2;         // 0‥3 → which 32-col band

        int warp_m = warp_row * 32;        // 0, 32, 64, 96
        int warp_n = warp_col * 32;        // 0, 32, 64, 96

        #pragma unroll
        for (int kk = 0; kk < CTA_K; kk += 16)
        {
            /* Load two 16×16 A fragments */
            wmma::load_matrix_sync(a[0], As + (warp_m + 0) * CTA_K + kk, CTA_K);
            wmma::load_matrix_sync(a[1], As + (warp_m + 16) * CTA_K + kk, CTA_K);

            /* Load two 16×16 B fragments */
            wmma::load_matrix_sync(b[0], Bs + (warp_n + 0) * CTA_K + kk, CTA_K);
            wmma::load_matrix_sync(b[1], Bs + (warp_n + 16) * CTA_K + kk, CTA_K);

            /* 4 MMAs   →   32 × 32 sub-tile of C */
            wmma::mma_sync(acc[0], a[0], b[0], acc[0]);   // (0,0)
            wmma::mma_sync(acc[1], a[0], b[1], acc[1]);   // (0,1)
            wmma::mma_sync(acc[2], a[1], b[0], acc[2]);   // (1,0)
            wmma::mma_sync(acc[3], a[1], b[1], acc[3]);   // (1,1)
        }
        __syncthreads();                       /* allow next K-slab to reuse SMEM */
    }

    // 5. guarded scatter to C
    /* One thread (lane 0) per warp scatters its 4 × 16 × 16 results. */
    int warp_id = threadIdx.x >> 5;
    int warp_row = warp_id & 3;          // 0‥3 → which 32-row band
    int warp_col = warp_id >> 2;         // 0‥3 → which 32-col band

    int warp_m = warp_row * 32;        // 0, 32, 64, 96
    int warp_n = warp_col * 32;        // 0, 32, 64, 96

    store_frag_as_fp16_guarded(acc[0], out,
                                cta_m + warp_m + 0,
                                cta_n + warp_n + 0,  M, N, smem);
    store_frag_as_fp16_guarded(acc[1], out,
                                cta_m + warp_m + 0,
                                cta_n + warp_n + 16, M, N, smem);
    store_frag_as_fp16_guarded(acc[2], out,
                                cta_m + warp_m + 16,
                                cta_n + warp_n + 0,  M, N, smem);
    store_frag_as_fp16_guarded(acc[3], out,
                                cta_m + warp_m + 16,
                                cta_n + warp_n + 16, M, N, smem);
}

/* =========================================================================
 *  Prefill Kernel Launcher
 *  Note that in prefill kernel, K is not divided by 32 originally
 * ========================================================================= */
void launch_prefill_sbvr_kernel(
    __nv_bfloat16* x,
    uint32_t* r_bvr, void* r_coeff_idx, __nv_bfloat16* r_coeff_cache,
    int r_num_sums,
    int r_cache_size,
    __nv_bfloat16* bias, __nv_bfloat16* out,
    int M, int N, int K)
{
    // 1. determine the type of r_coeff_idx
    const bool use_r_uint8 = (r_cache_size <= 256);

    // 2. set blockDim, gridDim and shared memory
    dim3 grid((N + CTA_N - 1) / CTA_N,
              (M + CTA_M - 1) / CTA_M);
    dim3 block(THREADS_PER_CTA);
    size_t smem_bytes = (CTA_M * CTA_K + CTA_K * CTA_N) * sizeof(half) + \
                        WARPS_PER_CTA * 16 * 16 * sizeof(float); // 16 + 16 KiB

    // 3. launch the kernel
    if (use_r_uint8) {
        switch (r_num_sums) {
            case 4:
                sbvr_prefill_mm_T<uint8_t, 4><<<grid, block, smem_bytes>>>(
                    x, 
                    r_bvr, (uint8_t*)r_coeff_idx, r_coeff_cache,
                    bias, out,
                    M, N, K);
                break;
            default:
                throw std::runtime_error("Unsupported r_num_sums value");
        }
    }
    else {
        switch (r_num_sums) {
            case 4:
                sbvr_prefill_mm_T<uint16_t, 4><<<grid, block, smem_bytes>>>(
                    x, 
                    r_bvr, (uint16_t*)r_coeff_idx, r_coeff_cache,
                    bias, out,
                    M, N, K);
                break;
            default:
                throw std::runtime_error("Unsupported r_num_sums value");
        }
    }
}