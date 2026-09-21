#include <torch/extension.h>

at::Tensor reduce_cuda(at::Tensor input);

at::Tensor reduce(at::Tensor input)
{
    return reduce_cuda(input);
}


TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("reduce", &reduce);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
// }

