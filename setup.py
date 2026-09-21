from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="kl",
    packages=["kl"],
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="kl._C",
            sources=[
                "src/ops/bindings.cpp",
                "src/ops/reduce.cpp",
                "src/cuda/reduce_kernel.cu",
                
                "src/ops/softmax.cpp",
                "src/cuda/softmax_kernel.cu",
                "src/ops/gemm4096.cpp",
                "src/cuda/gemm_kernel.cu",
                "src/ops/rmsnorm_add.cpp",
                "src/cuda/rmsnorm_add_kernel.cu",
                "src/ops/silu_x_mul.cpp",
                "src/cuda/silu_x_mul_kernel.cu",
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '-gencode=arch=compute_86,code=sm_86'] 
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)