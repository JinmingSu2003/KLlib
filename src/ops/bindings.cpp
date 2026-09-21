#include <torch/extension.h>

// ==========================================================
// 1. 前置声明 (Forward Declarations)
// 告诉编译器：这两个函数的具体实现写在了其他的 .cu 文件中
// ==========================================================
torch::Tensor softmax_cuda(torch::Tensor input);
torch::Tensor reduce_cuda(torch::Tensor input);
torch::Tensor gemm_cuda(
    torch::Tensor& A,
    torch::Tensor& B
);
torch::Tensor rmsnorm_add_cuda(torch::Tensor A,
torch::Tensor B,torch::Tensor C);
torch::Tensor siluxmul_cuda(torch::Tensor gate,
    torch::Tensor up);

// ==========================================================
// 2. 绑定模块 (Pybind11 Binding)
// ==========================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    // 可选：给你的模块写一段说明
    m.doc() = "KLlib CUDA Extension by glowing"; 

    // m.def 格式: 
    // m.def("Python中调用的函数名", &C++中的对应函数指针, "函数文档说明");
    
    m.def("softmax", &softmax_cuda, "Softmax forward (CUDA)");
    
    m.def("reduce", &reduce_cuda, "Reduce forward (CUDA)");

    m.def("gemm4096", &gemm_cuda, "gemm forward (CUDA)");
    m.def("rmsnorm_add", &rmsnorm_add_cuda, "rmsnorm_add forward (CUDA)");
    m.def("siluxmul", &siluxmul_cuda, "siluxmul forward (CUDA)");
}