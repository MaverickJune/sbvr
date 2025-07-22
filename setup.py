import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# read requirements…
with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip()]

here = os.path.dirname(__file__)
cutlass = os.path.join(here, "cutlass", "include")
sbvr_inc = os.path.join(here, "sbvr", "include")
fht_csrc = os.path.join(here, "third_party", "fast-hadamard-transform", "csrc")

setup(
    name="sbvr",
    version="0.1.0",
    packages=find_packages(include=[
        "sbvr", "sbvr.*",
        "fast_hadamard_transform", "fast_hadamard_transform.*",
    ]),
    package_dir={
        # point at the folder *containing* __init__.py
        "fast_hadamard_transform": os.path.join(
            "third_party", "fast-hadamard-transform", "fast_hadamard_transform"
        )
    },
    ext_modules=[
        CUDAExtension(
            name="sbvr.sbvr_cuda",
            sources=[
                "sbvr/kernels/sbvr_ops.cpp",
                "sbvr/kernels/sbvr_kernel.cu",
                "sbvr/kernels/rtn_sbvr_kernel.cu",
                "sbvr/kernels/sbvr_prefill_kernel.cu",
            ],
            include_dirs=[cutlass, sbvr_inc],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "--ftz=true", "-Xptxas=-v"],
            },
            extra_link_args=["-lcudadevrt"],
        ),
        CUDAExtension(
            name="fast_hadamard_transform.fast_hadamard_transform_cuda",
            sources=[
                os.path.join(fht_csrc, "fast_hadamard_transform.cpp"),
                os.path.join(fht_csrc, "fast_hadamard_transform_cuda.cu"),
            ],
            include_dirs=[fht_csrc],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=requirements,
    zip_safe=False,
)
