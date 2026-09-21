#include <torch/extension.h>

at::Tensor softmax_cuda(at::Tensor input);

at::Tensor softmax(at::Tensor input)
{
    return softmax_cuda(input);
}



TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("softmax", &softmax);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
// }