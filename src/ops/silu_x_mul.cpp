#include <torch/extension.h>

at::Tensor siluxmul_cuda(at::Tensor gate,
    at::Tensor up);
at::Tensor siluxmul(at::Tensor gate,
    at::Tensor up){
        return siluxmul_cuda(gate,up);
    }


TORCH_LIBRARY_IMPL(kl, CUDA, m)
{
    m.impl("siluxmul", &siluxmul);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
// }

