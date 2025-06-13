#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <assert.h>

// Declare the kernel launcher (templated in the actual .cu file)
void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx,__half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size,
    int device_id = 0);

// Declare the kernel launcher for row-wise, pre-dequantized SBVR (templated in the actual .cu file)
void launch_cuda_sbvr_row_deq_mm_T(
    __half* l_w, 
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int r_num_sums,
    int r_cache_size,
    int use_shfl = 0,
    int device_id = 0);

// Declare the kernel launcher for fused Rtn6 LUT BVR (templated in the actual .cu file)
void launch_fused_rtn_lut_bvr(
    const __half* x,
    uint32_t* out_bvr,
    float* scales,
    int num_groups = 128,
    int nRTN = 6);

int device_count;
cudaDeviceProp cuda_prop_list[16];

// PyTorch wrapper
torch::Tensor sbvr_mm_T(
                torch::Tensor l_bvr,
                torch::Tensor l_coeff_idx,
                torch::Tensor l_coeff_cache,
                torch::Tensor r_bvr,
                torch::Tensor r_coeff_idx,
                torch::Tensor r_coeff_cache,
                torch::Tensor bias
            )
{
    const int M = l_bvr.size(1);
    const int N = r_bvr.size(1);
    const int K = l_bvr.size(0);
    const int l_num_sums = l_bvr.size(2);
    const int r_num_sums = r_bvr.size(2);
    const int l_cache_size = l_coeff_cache.size(0);
    const int r_cache_size = r_coeff_cache.size(0);
    assert (l_bvr.size(0) == r_bvr.size(0));

    auto out = torch::empty({M, N},
                         torch::dtype(torch::kFloat16).device(l_bvr.device()));
    __half* bias_ptr = nullptr;
    if (bias.size(0) == N)
        bias_ptr = reinterpret_cast<__half*>(bias.data_ptr<at::Half>());
    

    // Call the dispatch kernel
    launch_cuda_sbvr_mm_T(
        l_bvr.data_ptr<uint32_t>(),
        l_coeff_idx.data_ptr(),  // dispatch will cast correctly
        reinterpret_cast<__half*>(l_coeff_cache.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint32_t>(),
        r_coeff_idx.data_ptr(),
        reinterpret_cast<__half*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        M, N, K,
        l_num_sums, r_num_sums,
        l_cache_size, r_cache_size);

    return out;
}

// PyTorch wrapper for row-wise, pre-dequnatize SBVR 
torch::Tensor sbvr_row_deq_mm_T(
                torch::Tensor l_w,
                torch::Tensor r_bvr,
                torch::Tensor r_coeff_idx,
                torch::Tensor r_coeff_cache,
                torch::Tensor bias,
                int use_shfl = 0
            )
{
    /* 
    C = A @ B^T
    r_bvr is grouped in row-direction, (N/32(num_bits in bvr dtype, uint32), K, num_sums)
    */

    const int M = l_w.size(0);
    const int N = r_bvr.size(0) * 32;
    const int K = r_bvr.size(1);
    const int r_num_sums = r_bvr.size(2);
    const int r_cache_size = r_coeff_cache.size(0);
    assert (l_w.size(1) == K);

    auto out = torch::empty({M, N},
                         torch::dtype(torch::kFloat16).device(l_w.device()));
    __half* bias_ptr = nullptr;
    if (bias.size(0) == N)
        bias_ptr = reinterpret_cast<__half*>(bias.data_ptr<at::Half>());

    // Call the dispatch kernel
    launch_cuda_sbvr_row_deq_mm_T(
        reinterpret_cast<__half*>(l_w.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint32_t>(),
        r_coeff_idx.data_ptr(),
        reinterpret_cast<__half*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        M, N, K,
        r_num_sums,
        r_cache_size,
        use_shfl);

    return out;
}

// PyTorch wrapper for sbvr input transform
torch::Tensor sbvr_input_transfrom(
    torch::Tensor x,
    int nRTN = 6,
    int group_size = 128)
{
    const int M = x.size(0);
    const int K = x.size(1);
    const int _nRTN = nRTN + 2;
    const int c_ratio = 32 / _nRTN;
    const int num_groups = K / group_size;

    auto out_bvr = torch::empty({K / c_ratio}, torch::dtype(torch::kUInt32).device(x.device()));
    auto scales = torch::empty({K / group_size}, torch::dtype(torch::kFloat32).device(x.device()));

    launch_fused_rtn_lut_bvr(
        reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
        out_bvr.data_ptr<uint32_t>(),
        scales.data_ptr<float>(),
        num_groups,
        nRTN
    );

    return (out_bvr, scales);
}

void sbvr_cuda_init() 
{
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) 
    {
        std::cerr << "\033[91mSBVR Init:\033[0m No CUDA devices found: " << 
            cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "\033[92mSBVR Init:\033[0m Found " << device_count 
              << " CUDA device(s)." << std::endl;
    for (int device_id = 0; device_id < device_count; ++device_id) 
    {
        err = cudaGetDeviceProperties(&cuda_prop_list[device_id], device_id);
        auto prop = cuda_prop_list[device_id];
        if (err != cudaSuccess) 
        {
            std::cerr << "\tFailed to get properties for device " << device_id 
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "\tDevice " << device_id << ": " << prop.name 
                  << " (Compute Capability: " << prop.major << "." 
                  << prop.minor << ")" << std::endl;
    }
    std::cout << "\033[92mSBVR Init:\033[0m" 
              << " CUDA Initialization complete." << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_sbvr_cuda_init", &sbvr_cuda_init,
        "Initialization function for the SBVR CUDA kernels");
    m.def("_sbvr_mm_T", &sbvr_mm_T, 
          py::arg("l_bvr"),
          py::arg("l_coeff_idx"),
          py::arg("l_coeff_cache"),
          py::arg("r_bvr"),
          py::arg("r_coeff_idx"),
          py::arg("r_coeff_cache"),
          py::arg("bias"),
          "SBVR Matrix-Matrix_Transposed Multiplication kernel");
    m.def("_sbvr_row_deq_mm_T", &sbvr_row_deq_mm_T,
          py::arg("l_w"),
          py::arg("r_bvr"),
          py::arg("r_coeff_idx"),
          py::arg("r_coeff_cache"),
          py::arg("bias"),
          py::arg("use_shfl") = 0,
          "SBVR Row-wise, pre-dequantized Matrix-Matrix_Transposed Multiplication kernel");
    m.def("_sbvr_input_transfrom", &sbvr_input_transfrom,
          py::arg("x"),
          py::arg("nRTN") = 6,
          py::arg("group_size") = 128,
          "SBVR Input Transform kernel");
}