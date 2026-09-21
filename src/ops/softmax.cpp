#include <torch/extension.h>

at::Tensor softmax_cuda(at::Tensor input);

// schema 统一在 bindings.cpp 的 TORCH_LIBRARY(kl, m) 里声明
TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("softmax", &softmax_cuda);
}
