#include"cuda_runtime.h"
#include<cstdio>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define FLOAT4(ptr) (*reinterpret_cast<float4*> (ptr))


__device__ float warp_shuffle(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val+= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


template<int M,int N,int warp_num>
__global__ void rmsnorm_add_kernel(
    float* A,
    float* B,
    float* W,
    float* O
)
{
    float eps=1e-6;
    const int row=blockIdx.x;
    const int tid=threadIdx.x;
    const int warp_id=tid/32;
    const int lane_id=tid%32;
    __shared__ float rowsum;
    __shared__ float smem[warp_num];
    float acc=0.0f;
    for(int i=tid;i<N/4;i=i+blockDim.x)
    {
        float4 a=FLOAT4(A+row*N+i*4);
        float4 b=FLOAT4(B+row*N+i*4);
        acc+=((a.x+b.x)*(a.x+b.x)+(a.y+b.y)*(a.y+b.y)
        +(a.z+b.z)*(a.z+b.z)+(a.w+b.w)*(a.w+b.w));
    }
    __syncthreads();
    acc=warp_shuffle(acc);
    if(lane_id==0)smem[warp_id]=acc;
    __syncthreads();
    if(warp_id==0)
    {
        acc=lane_id<warp_num?smem[lane_id]:0.0f;
        acc=warp_shuffle(acc);
        if(lane_id==0)rowsum=rsqrt(acc/N+eps);
    }
    __syncthreads();

    for(int i=tid;i<N/4;i+=blockDim.x)
    {
        float4 a=FLOAT4(A+row*N+i*4);
        float4 w=FLOAT4(W+i*4);
        float4 b=FLOAT4(B+row*N+i*4);
        float4 r=make_float4((a.x+b.x)*rowsum*w.x,
                             (a.y+b.y)*rowsum*w.y,
                            (a.z+b.z)*rowsum*w.z,
                        (a.w+b.w)*rowsum*w.w);
        FLOAT4(O+row*N+4*i)=r;
    }
}


at::Tensor rmsnorm_add_cuda(at::Tensor A,
    at::Tensor B,at::Tensor W)
{
    const at::cuda::OptionalCUDAGuard guard(A.device());
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto W_c = W.contiguous();
    auto output = at::empty_like(A_c);
   
    dim3 gridsize(4096);
    dim3 blocksize(128);
    constexpr int M=4096;
    constexpr int N=4096;
    constexpr int warp=4;
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_add_kernel<M,N,warp><<<gridsize,blocksize>>>(A_c.data_ptr<float>(),
    B_c.data_ptr<float>(),
    W_c.data_ptr<float>(),
    output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// int main()
// {

//     constexpr int M=4096,N=4096;
//     float* a=(float*)malloc(M*N*sizeof(float));
//         float* o=(float*)malloc(M*N*sizeof(float));
//     float* b=(float*)malloc(M*N*sizeof(float));
//     float* w=(float*)malloc(N*sizeof(float));
//     for(int i=0;i<M*N;i++)
//     {
//         a[i]=1;
//         b[i]=2;
//     }
//     for(int i=0;i<N;i++)w[i]=3;
//     float* a_d;
//     float* b_d;
//     float* w_d;
//     float* o_d;
//     cudaMalloc(&a_d,M*N*sizeof(float));
//     cudaMalloc(&b_d,M*N*sizeof(float));
//     cudaMalloc(&w_d,N*sizeof(float));
//     cudaMalloc(&o_d,M*N*sizeof(float));
//     cudaMemcpy(a_d,a,M*N*sizeof(float),cudaMemcpyHostToDevice);
//     cudaMemcpy(b_d,b,M*N*sizeof(float),cudaMemcpyHostToDevice);
//     cudaMemcpy(w_d,w,N*sizeof(float),cudaMemcpyHostToDevice);
//     dim3 blocksize(128);
//     dim3 gridsize(4096);
//     constexpr int warpnum=128/32;
//     rmsnorm_add_kernel<M,N,warpnum><<<gridsize,blocksize>>>(a_d,b_d,w_d,o_d);
//      cudaDeviceSynchronize();
//     cudaMemcpy(o, o_d, M*N*sizeof(float),cudaMemcpyDeviceToHost);
//     printf("output[0] = %f\n", o[0]);
//     return 0;
// }