#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdio>

#define FLOAT4(ptr) (*reinterpret_cast<float4*>(ptr))
#define CONST_FLOAT4(ptr) (*reinterpret_cast<const float4*>(ptr))


__global__ void silu_x_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int N
) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int N4 = N / 4;

    // 每个线程处理一个 float4，即连续4个FP32元素
    for (int idx4 = tid; idx4 < N4; idx4 += stride) {
        const int offset = idx4 * 4;

        float4 g = CONST_FLOAT4(gate + offset);
        float4 u = CONST_FLOAT4(up + offset);

        float4 result;

        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;

        FLOAT4(out + offset) = result;
    }
}


at::Tensor siluxmul_cuda(at::Tensor gate,
    at::Tensor up)
{
    const at::cuda::OptionalCUDAGuard guard(gate.device());

    auto gate_c = gate.contiguous();
    auto up_c = up.contiguous();
    auto output = at::empty_like(gate_c);

    int n=gate_c.numel();
    // int row=gate_c.size(0);
    // int col=gate_c.size(1);
    dim3 blocksize(256);
    dim3 gridsize((n+255)/blocksize.x);
    auto stream = at::cuda::getCurrentCUDAStream();
    silu_x_mul_kernel<<<gridsize,blocksize>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


