#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#define FLOAT4(ptr) (*reinterpret_cast<float4*>(ptr))
template<int M,int N,int K>
__global__ void gemm_naive(
    float*A,float*B,float*C
)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    const int col=blockIdx.y*blockDim.y+threadIdx.y;
    float* a_start=A+row*K;
    float* b_start=B+col;
    float acc=0.0f;
    for(int i=0;i<K;i++)
    {
        acc+=a_start[i]*b_start[i*N];
    }
    if(row<M&&col<N)
    {
        C[row*N+col]=acc;
    }
}

template<int M,int N,int K,int BLOCK_M,int BLOCK_N>
__global__ void gemm_shared_float4(
    float* A,float* B,float* C
)
{
    int tid=threadIdx.x;
    int block_col=blockIdx.x;
    int block_row=blockIdx.y;
    __shared__ float smem_a[BLOCK_M*K];
    __shared__ float smem_b[K*BLOCK_N];
    float* a_start=A+block_row*BLOCK_M*K;
    float* b_start=B+block_col*BLOCK_N;
    if(tid>=K)return;
    #pragma unroll
    for(int i=0;i<K;i+=4)
    {
        if(tid<BLOCK_M)
        {
            float4 x=FLOAT4(a_start+tid*K+i);
            smem_a[tid*K+i]=x.x;
            smem_a[tid*K+i+1]=x.y;
            smem_a[tid*K+i+2]=x.z;
            smem_a[tid*K+i+3]=x.w;
        }

        if (tid < BLOCK_N)
        {
             smem_b[tid*K+i]=b_start[i*N+tid];
             smem_b[tid*K+i+1]=b_start[(i+1)*N+tid];
             smem_b[tid*K+i+2]=b_start[(i+2)*N+tid];
             smem_b[tid*K+i+3]=b_start[(i+3)*N+tid];
        }
       
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<BLOCK_N;i++)
    {
        float acc=0.0f;
        for(int j=0;j<K;j++)
        {
            if (tid < BLOCK_M) acc+=smem_a[tid*K+j]*smem_b[i*K+j];
        }
       if (tid < BLOCK_M)C[(block_row*BLOCK_M+tid)*N+block_col*BLOCK_N+i]=acc;     
    }
}


template <
    int M, int N, int K,
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int TM, int TN
>
__global__ void gemm_register_tile(
    float* A, float* B, float* C
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // A tile 为 64×16，每行由 4 个线程搬运。
    const int a_block_row = tid / 4;
    const int a_block_col = (tid % 4) * 4;

    // B tile 为 16×64，每行由 16 个线程搬运。
    const int b_block_row = tid / 16;
    const int b_block_col = (tid % 16) * 4;

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    __shared__ float smem_a[BLOCK_M][BLOCK_K];
    __shared__ float smem_b[BLOCK_K][BLOCK_N];

    float* a_start = A + block_row * BLOCK_M * K;
    float* b_start = B + block_col * BLOCK_N;

    // 当前线程负责的输出块，在共享 tile 内的起点。
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    float acc[TM][TN] = {0.0f};

    float reg_a[TM];
    float reg_b[TN];

    // 不展开整个 K 维大循环，只展开内部小循环。
    for (int k_s = 0; k_s < K; k_s += BLOCK_K) {

        // 1. 全局内存 → 共享内存
        float4 a = FLOAT4(
            a_start + a_block_row * K + a_block_col + k_s
        );

        smem_a[a_block_row][a_block_col]     = a.x;
        smem_a[a_block_row][a_block_col + 1] = a.y;
        smem_a[a_block_row][a_block_col + 2] = a.z;
        smem_a[a_block_row][a_block_col + 3] = a.w;

        float4 b = FLOAT4(
            b_start + (b_block_row + k_s) * N + b_block_col
        );

        smem_b[b_block_row][b_block_col]     = b.x;
        smem_b[b_block_row][b_block_col + 1] = b.y;
        smem_b[b_block_row][b_block_col + 2] = b.z;
        smem_b[b_block_row][b_block_col + 3] = b.w;

        __syncthreads();

        // 2. 遍历当前 tile 的 K 维，每轮做一次外积。
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            // A 取一列中的 TM 个元素。
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = smem_a[thread_row + i][kk];
            }
            // B 取一行中的 TN 个元素。
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = smem_b[kk][thread_col + j];
            }
            // 外积累加：TM×1 乘 1×TN。
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(
                        reg_a[i], reg_b[j], acc[i][j]
                    );
                }
            }
        }

        // 读完本轮共享内存后，才能搬运下一轮。
        __syncthreads();
    }

    // 3. 寄存器 → 全局内存
    const int c_row = block_row * BLOCK_M + thread_row;
    const int c_col = block_col * BLOCK_N + thread_col;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(c_row + i) * N + c_col + j] = acc[i][j];
        }
    }
}

template<int M,int N,int K,
         int BLOCK_M,int BLOCK_N,int BLOCK_K,
         int TM,int TN>
__global__ void gemm(//m128 n64 k16 tm8 tn8 4096 4096 blocksize 128
    float*  A,
    float*  B,
    float*  C
)
{
    const int tid=threadIdx.x;
    const int block_row=blockIdx.y*BLOCK_M;
    const int block_col=blockIdx.x*BLOCK_N;
    float* a_start=A+block_row*K;
    float* b_start=B+block_col;
    const int warp_id=tid/32;
    const int lane_id=tid%32;
    //128 64
    //为了合并访存 warp形状是4*8
    const int row=warp_id*32+(lane_id/8)*4;
    const int col=(lane_id%8)*4;
    __shared__ float smem_a[BLOCK_K][BLOCK_M+4];
    __shared__ float smem_b[BLOCK_K][BLOCK_N];
    float acc[TM][TN]={0.0f};
    for(int k_s=0;k_s<K;k_s+=BLOCK_K)
    {
        for(int i=0;i<4;i++)
        {
            float4 ar=FLOAT4(a_start+tid*K+k_s+4*i);
            smem_a[4*i][tid]=ar.x;
            smem_a[4*i+1][tid]=ar.y;
            smem_a[4*i+2][tid]=ar.z;
            smem_a[4*i+3][tid]=ar.w;     
        }
        int b_row=tid/16;
        int b_col=(tid%16)*4;
        for(int i=0;i<2;i++)
        {
            int b_row1=b_row+8*i;
            float4 br=FLOAT4(b_start+k_s*N+b_row1*N+b_col);
            smem_b[b_row1][b_col]=br.x;
            smem_b[b_row1][b_col+1]=br.y;
            smem_b[b_row1][b_col+2]=br.z;
            smem_b[b_row1][b_col+3]=br.w;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BLOCK_K;k++)
        {
            float4 a0=FLOAT4(&smem_a[k][row]);
            float4 a1=FLOAT4(&smem_a[k][row+16]);
            float4 b0=FLOAT4(&smem_b[k][col]);
            float4 b1=FLOAT4(&smem_b[k][col+32]);
            float ar[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
            float br[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
            #pragma unroll
            for(int i=0;i<8;++i) 
            {
                #pragma unroll
                for(int j=0;j<8;++j)
                    acc[i][j]=fmaf(ar[i],br[j],acc[i][j]);
            }
        }
        __syncthreads();

    }

    #pragma unroll
    for(int i=0;i<8;i++)
    {
        int c_row = row + (i < 4 ? i : i + 12);
        float* c_start =C + (block_row + c_row) * N+ block_col + col;
        float4 c1=make_float4(acc[i][0],acc[i][1],acc[i][2],acc[i][3]);
        float4 c2=make_float4(acc[i][4],acc[i][5],acc[i][6],acc[i][7]);
        FLOAT4(c_start)=c1;
        FLOAT4(c_start+32)=c2;
    }
    
    
}


at::Tensor gemm_cuda(
    at::Tensor& A,
    at::Tensor& B
) {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 16;

    constexpr int TM = 8;
    constexpr int TN = 8;

    TORCH_CHECK(A.is_cuda(),
                "gemm4096: A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),
                "gemm4096: B must be a CUDA tensor");

    TORCH_CHECK(A.scalar_type() == at::kFloat,
                "gemm4096: A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat,
                "gemm4096: B must be float32");

    TORCH_CHECK(A.dim() == 2,
                "gemm4096: A must be a 2D matrix");
    TORCH_CHECK(B.dim() == 2,
                "gemm4096: B must be a 2D matrix");

    TORCH_CHECK(
        A.size(0) == M && A.size(1) == K,
        "gemm4096: A must have shape [4096, 4096], but got ",
        A.sizes()
    );

    TORCH_CHECK(
        B.size(0) == K && B.size(1) == N,
        "gemm4096: B must have shape [4096, 4096], but got ",
        B.sizes()
    );

    TORCH_CHECK(
        A.device() == B.device(),
        "gemm4096: A and B must be on the same CUDA device"
    );

    // 保证接下来的分配和 kernel 都在 A 所在设备执行
    at::cuda::OptionalCUDAGuard device_guard(A.device());

    // 你的 kernel 假设矩阵为 row-major contiguous
    auto A_contiguous = A.contiguous();
    auto B_contiguous = B.contiguous();

    auto output = at::empty(
        {M, N},
        A_contiguous.options()
    );

    const dim3 grid(
        N / BLOCK_N,   // 4096 / 64  = 64
        M / BLOCK_M    // 4096 / 128 = 32
    );

    const dim3 block(128);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gemm<
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        TM, TN
    ><<<grid, block, 0, stream>>>(
        A_contiguous.data_ptr<float>(),
        B_contiguous.data_ptr<float>(),
        output.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}