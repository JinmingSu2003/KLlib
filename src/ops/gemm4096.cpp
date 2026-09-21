#include <torch/extension.h>

// CUDA 实现在 .cu 文件中
at::Tensor gemm_cuda(
    at::Tensor& A,
    at::Tensor& B
);

at::Tensor gemm4096(
    at::Tensor& A,
    at::Tensor& B
) {
    return gemm_cuda(A, B);
}

TORCH_LIBRARY_IMPL(kl, CUDA, m) {
    m.impl("gemm4096", &gemm4096);
}
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
// }


