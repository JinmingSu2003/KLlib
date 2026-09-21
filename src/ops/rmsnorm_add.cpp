#include <torch/extension.h>

at::Tensor rmsnorm_add_cuda(at::Tensor A,at::Tensor B,at::Tensor W);

at::Tensor rmsnorm_add(at::Tensor A,at::Tensor B,at::Tensor W)
{
    return rmsnorm_add_cuda(A,B,W);
}


TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("rmsnorm_add", &rmsnorm_add);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
// }

