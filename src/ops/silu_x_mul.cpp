#include <torch/extension.h>

at::Tensor siluxmul_cuda(at::Tensor gate,
    at::Tensor up);

// schema 统一在 bindings.cpp 的 TORCH_LIBRARY(kl, m) 里声明
TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("siluxmul", &siluxmul_cuda);
}


