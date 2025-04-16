
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

__global__ void naive_sbvr_mm(uint32_t* l_bvr,
                              uint8_t* l_coeff_idx,
                              uint8_t* l_bias_idx,
                              __half* l_coeff_cache,
                              __half* l_bias_cache,
                              uint32_t* r_bvr,
                              uint8_t* r_coeff_idx,
                              uint8_t* r_bias_idx,
                              __half* r_coeff_cache,
                              __half* r_bias_cache,
                              __half* out,
                              int out_rows,
                              int out_cols,
                              int num_sums,
                              int cgroup_per_inner_vec,
                              int bvr_per_cgroup,
                              int cache_size,
                              int num_cgroups) 
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
}

extern "C" void launch_sbvr_mm(
                    uint32_t* l_bvr,
                    uint8_t* l_coeff_idx,
                    uint8_t* l_bias_idx,
                    __half* l_coeff_cache,
                    __half* l_bias_cache,
                    uint32_t* r_bvr,
                    uint8_t* r_coeff_idx,
                    uint8_t* r_bias_idx,
                    __half* r_coeff_cache,
                    __half* r_bias_cache,
                    __half* out,
                    int out_rows,
                    int out_cols,
                    int num_sums,
                    int cgroup_per_inner_vec,
                    int bvr_per_cgroup,
                    int cache_size,
                    int num_cgroups) 
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
    dim3 threads = 256;
    std::cout << "Blocks: " << blocks << ", Threads: " << threads.x << std::endl;
    naive_sbvr_mm<<<blocks, threads>>>(l_bvr,
                                       l_coeff_idx,
                                       l_bias_idx,
                                       l_coeff_cache,
                                       l_bias_cache,
                                       r_bvr,
                                       r_coeff_idx,
                                       r_bias_idx,
                                       r_coeff_cache,
                                       r_bias_cache,
                                       out,
                                       out_rows,
                                       out_cols,
                                       num_sums,
                                       cgroup_per_inner_vec,
                                       bvr_per_cgroup,
                                       cache_size,
                                       num_cgroups);
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}
