#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// 每个 block 负责一行，沿最后一维做求和归约。
// 先用 grid-stride 循环做局部累加，再做 warp + shared memory 树状归约。

#define REDUCE_BLOCK_SIZE 512

__device__ __forceinline__ float warpReduceSum(float sum) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    return sum;
}

__global__ void reduce_rows_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   long n) {
    const long row = blockIdx.x;
    const float* row_ptr = in + row * n;

    float sum = 0.f;
    for (long i = threadIdx.x; i < n; i += blockDim.x) {
        sum += row_ptr[i];
    }

    __shared__ float shared[REDUCE_BLOCK_SIZE];
    const unsigned lane = threadIdx.x & 31;
    const unsigned wid = threadIdx.x >> 5;
    const unsigned numWarps = REDUCE_BLOCK_SIZE / 32;

    sum = warpReduceSum(sum);
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (lane < numWarps) ? shared[lane] : 0.f;
        sum = warpReduceSum(sum);
        if (lane == 0) out[row] = sum;
    }
}



at::Tensor reduce_cuda(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "reduce: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "reduce: input must be float32");
    TORCH_CHECK(input.dim() >= 1, "reduce: input must have at least 1 dimension");
    TORCH_CHECK(input.numel() > 0, "reduce: input must be non-empty");

    const at::cuda::OptionalCUDAGuard guard(input.device());
    auto input_c = input.contiguous();

    const long n = input_c.size(-1);
    const long rows = input_c.numel() / n;

    // 输出形状 = 去掉最后一维，1D 输入退化为 0 维标量（与 torch.sum 一致）
    std::vector<int64_t> out_shape(input_c.sizes().begin(),
                                   input_c.sizes().end() - 1);
    auto output = at::empty(out_shape, input_c.options());

    if (rows > 0 && n > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        reduce_rows_kernel<<<(unsigned)rows, REDUCE_BLOCK_SIZE, 0, stream>>>(
            input_c.data_ptr<float>(), output.data_ptr<float>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return output;
}
