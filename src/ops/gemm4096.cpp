#include <torch/extension.h>

at::Tensor gemm_cuda(
    at::Tensor A,
    at::Tensor B
);

// schema 统一在 bindings.cpp 的 TORCH_LIBRARY(kl, m) 里声明
TORCH_LIBRARY_IMPL(kl, CUDA, m) {
    m.impl("gemm4096", &gemm_cuda);
}



