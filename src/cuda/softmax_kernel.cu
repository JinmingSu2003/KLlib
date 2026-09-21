#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cfloat>

#define SOFTMAX_BLOCK_SIZE 256

__device__ float warp_reduce_max(float val)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1)
    {
        val = fmaxf(
            val,
            __shfl_down_sync(0xffffffff, val, offset)
        );
    }

    return val;
}


__device__ float warp_reduce_sum(float val)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}


template<int BLOCKSIZE>
__device__ float block_reduce_max(float val)
{
    __shared__ float maxx[32];
    __shared__ float blockmax;

    int warpid = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (laneid == 0)
    {
        maxx[warpid] = val;
    }

    __syncthreads();

    if (warpid == 0)
    {
        float ans = (laneid < BLOCKSIZE / 32)
                        ? maxx[laneid]
                        : -INFINITY;
        ans = warp_reduce_max(ans);
        if (laneid == 0)
        {
            blockmax = ans;
        }
    }

    // 所有线程都要读 blockmax，必须同步。
    __syncthreads();

    return blockmax;
}


template<int BLOCKSIZE>
__device__ float block_reduce_sum(float val)
{
    __shared__ float sums[32];
    __shared__ float blocksum;

    int warpid = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (laneid == 0)
    {
        sums[warpid] = val;
    }

    __syncthreads();

    if (warpid == 0)
    {
        // 关键：不足的位置补 0，否则会污染求和结果。
        float ans = (laneid < BLOCKSIZE / 32)
                        ? sums[laneid]
                        : 0.0f;
        ans = warp_reduce_sum(ans);
        if (laneid == 0)
        {
            blocksum = ans;
        }
    }

    // 所有线程都要读 blocksum，必须同步。
    __syncthreads();

    return blocksum;
}


template <unsigned int BLOCK_SIZE>
__global__ void softmax_rows_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    long n) 
{
    const long row=blockIdx.x;
    const float* row_start=in+n*row;
    float* row_output=out+n*row;
    float old_m=-INFINITY;
    float m=-INFINITY;
    float s=0.0f;
    for(long i=threadIdx.x;i<n;i+=blockDim.x)
    {
        float x=row_start[i];
        old_m=m;
        m=fmaxf(m,x);
        s=expf(x-m)+s*expf(old_m-m);
    }
    __syncthreads();

    float maxx=block_reduce_max<BLOCK_SIZE>(m);
    float sumx=block_reduce_sum<BLOCK_SIZE>(s);

    float inv=1.0f/sumx;
    for(long i=threadIdx.x;i<n;i+=blockDim.x)
    {
        row_output[i]=expf(row_start[i]-maxx)*inv;
    }
}

at::Tensor softmax_cuda(at::Tensor input) {
    const at::cuda::OptionalCUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const long n = input_c.size(-1);
    const long rows = input_c.numel() / n;
    auto output = at::empty_like(input_c);
    if (rows > 0 && n > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        softmax_rows_kernel<SOFTMAX_BLOCK_SIZE><<<(unsigned)rows, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
            input_c.data_ptr<float>(), output.data_ptr<float>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return output;
}
