
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

template <typename LIndexT, typename RIndexT>
__global__ void cuda_naive_sbvr_mm_T(
    uint32_t* l_bvr,
    LIndexT* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    RIndexT* r_coeff_idx,
    __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int out_rows,
    int out_cols,
    int l_num_sums,
    int r_num_sums,
    int cgroup_per_inner_vec,
    int bvr_per_cgroup)
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]

    float coeff_mult[10][10];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = out_rows * out_cols;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        int row = i / out_cols;
        int col = i % out_cols;
        float sum = (bias != nullptr) ? __half2float(bias[col]) : 0.0f;

        // if (tid == 0)
        //     printf("Tid %d (%d, %d), bias: %f, out_rows: %d, out_cols: %d, "
        //         "l_num_sums: %d, r_num_sums: %d, cgroup_per_inner_vec: %d, "
        //         "bvr_per_cgroup: %d\n",
        //         tid, row, col, sum, out_rows, out_cols, 
        //         l_num_sums, r_num_sums, cgroup_per_inner_vec,
        //         bvr_per_cgroup);

        for (int cg_idx = 0; cg_idx < cgroup_per_inner_vec; cg_idx++)
        {
            int l_idx_flat = row * cgroup_per_inner_vec + cg_idx;
            int r_idx_flat = col * cgroup_per_inner_vec + cg_idx;
            int l_coeff_cache_idx = l_coeff_idx[l_idx_flat];
            int r_coeff_cache_idx = r_coeff_idx[r_idx_flat];
            __half* l_coeff_ptr = 
                            &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half* r_coeff_ptr = 
                            &r_coeff_cache[r_coeff_cache_idx * r_num_sums];

            // Precompute the coefficient multiplications
            for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
            {
                float l_coeff = __half2float(l_coeff_ptr[l_idx]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    coeff_mult[l_idx][r_idx] = 
                        l_coeff * __half2float(r_coeff_ptr[r_idx]);
                    // if (tid == 0)
                    //     printf("cg_idx: %d, l_coeff: %f, "
                    //         "r_coeff: %f, coeff_mult: %f\n", 
                    //         cg_idx, l_coeff, 
                    //         __half2float(r_coeff_ptr[r_idx]), 
                    //         l_coeff * __half2float(r_coeff_ptr[r_idx]));
                }
            }

            for (int bvr_idx = 0; bvr_idx < bvr_per_cgroup; bvr_idx++)
            {
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    uint32_t l = l_bvr[
                        row * cgroup_per_inner_vec * 
                        bvr_per_cgroup * l_num_sums +
                        l_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        uint32_t r = r_bvr[
                            col * cgroup_per_inner_vec * 
                            bvr_per_cgroup * r_num_sums +
                            r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                            cg_idx * bvr_per_cgroup + bvr_idx];
                        uint32_t lr = l & r;
                        float lr_popc = (float)__popc(lr);
                        sum += lr_popc * coeff_mult[l_idx][r_idx];
                        // if (tid == 0)
                        //     printf("bvr_idx: %d, l: %u, r: %u, lr: %u, "
                        //         "lr_popc: %f, coeff_mult: %f, sum: %f\n", 
                        //         bvr_idx, l, r, lr, lr_popc, 
                        //         coeff_mult[l_idx][r_idx], sum);
                    }
                }
            }

        }

        // Store the result in the output matrix
        out[i] = __float2half(sum);
    }
    
}

template <typename LIndexT, typename RIndexT>
__global__ void cuda_4x8_sbvr_mm_T(
    uint32_t* l_bvr,
    LIndexT* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    RIndexT* r_coeff_idx,
    __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int out_rows,
    int out_cols,
    int l_num_sums,
    int r_num_sums,
    int cgroup_per_inner_vec,
    int bvr_per_cgroup)
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]

    float coeff_mult[10][10];  // max size constraint (could be dynamic)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = out_rows * out_cols;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        int row = i / out_cols;
        int col = i % out_cols;
        float sum = (bias != nullptr) ? __half2float(bias[col]) : 0.0f;

        for (int cg_idx = 0; cg_idx < cgroup_per_inner_vec; cg_idx++) 
        {
            int l_idx_flat = row * cgroup_per_inner_vec + cg_idx;
            int r_idx_flat = col * cgroup_per_inner_vec + cg_idx;

            int l_coeff_cache_idx = l_coeff_idx[l_idx_flat];
            int r_coeff_cache_idx = r_coeff_idx[r_idx_flat];

            __half* l_coeff_ptr = &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half* r_coeff_ptr = &r_coeff_cache[r_coeff_cache_idx * r_num_sums];

            for (int l_idx = 0; l_idx < l_num_sums; l_idx++) {
                float l_coeff = __half2float(l_coeff_ptr[l_idx]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++) {
                    coeff_mult[l_idx][r_idx] = l_coeff * __half2float(r_coeff_ptr[r_idx]);
                }
            }

            // Do something with coeff_mult[l_idx][r_idx] if needed
        }

        // Write final sum if needed
        // out[row * out_cols + col] = __float2half(sum);
    }
}


template <typename LIndexT, typename RIndexT>
void launch_naive_sbvr_kernel(
    uint32_t* l_bvr,
    LIndexT* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    RIndexT* r_coeff_idx,
    __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int out_rows,
    int out_cols,
    int l_num_sums,
    int r_num_sums,
    int cgroup_per_inner_vec,
    int bvr_per_cgroup)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 8;
    dim3 threads = 32;

    cuda_naive_sbvr_mm_T<LIndexT, RIndexT><<<blocks, threads>>>(
        l_bvr,
        l_coeff_idx,
        l_coeff_cache,
        r_bvr,
        r_coeff_idx,
        r_coeff_cache,
        bias,
        out,
        out_rows,
        out_cols,
        l_num_sums,
        r_num_sums,
        cgroup_per_inner_vec,
        bvr_per_cgroup
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr,
    void* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    void* r_coeff_idx,
    __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int out_rows,
    int out_cols,
    int l_num_sums,
    int r_num_sums,
    int l_cache_size,
    int r_cache_size,
    int cgroup_per_inner_vec,
    int bvr_per_cgroup)
{
    bool use_l_uint16 = (l_cache_size > 256);
    bool use_r_uint16 = (r_cache_size > 256);

    if (!use_l_uint16 && !use_r_uint16) {
        launch_naive_sbvr_kernel<uint8_t, uint8_t>(
            l_bvr, (uint8_t*)l_coeff_idx, l_coeff_cache,
            r_bvr, (uint8_t*)r_coeff_idx, r_coeff_cache,
            bias, out,
            out_rows, out_cols,
            l_num_sums, r_num_sums,
            cgroup_per_inner_vec, bvr_per_cgroup);
    } else if (use_l_uint16 && !use_r_uint16) {
        launch_naive_sbvr_kernel<uint16_t, uint8_t>(
            l_bvr, (uint16_t*)l_coeff_idx, l_coeff_cache,
            r_bvr, (uint8_t*)r_coeff_idx, r_coeff_cache,
            bias, out,
            out_rows, out_cols,
            l_num_sums, r_num_sums,
            cgroup_per_inner_vec, bvr_per_cgroup);
    } else if (!use_l_uint16 && use_r_uint16) {
        launch_naive_sbvr_kernel<uint8_t, uint16_t>(
            l_bvr, (uint8_t*)l_coeff_idx, l_coeff_cache,
            r_bvr, (uint16_t*)r_coeff_idx, r_coeff_cache,
            bias, out,
            out_rows, out_cols,
            l_num_sums, r_num_sums,
            cgroup_per_inner_vec, bvr_per_cgroup);
    } else {
        launch_naive_sbvr_kernel<uint16_t, uint16_t>(
            l_bvr, (uint16_t*)l_coeff_idx, l_coeff_cache,
            r_bvr, (uint16_t*)r_coeff_idx, r_coeff_cache,
            bias, out,
            out_rows, out_cols,
            l_num_sums, r_num_sums,
            cgroup_per_inner_vec, bvr_per_cgroup);
    }
}