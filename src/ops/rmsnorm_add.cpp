#include <torch/extension.h>

at::Tensor rmsnorm_add_cuda(at::Tensor A,at::Tensor B,at::Tensor W);

// schema 统一在 bindings.cpp 的 TORCH_LIBRARY(kl, m) 里声明
TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("rmsnorm_add", &rmsnorm_add_cuda);
}


