#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define FLOAT4(ptr) (*reinterpret_cast<float4*>(ptr))

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

#define CUBLAS_CHECK(call)                                                  \
do {                                                                       \
    cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",                \
                __FILE__, __LINE__, static_cast<int>(status));              \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)


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


int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 16;

    constexpr int TM = 8;
    constexpr int TN = 8;

    constexpr int WARMUP = 10;
    constexpr int REPEAT = 100;

    static_assert(M % BLOCK_M == 0);
    static_assert(N % BLOCK_N == 0);
    static_assert(K % BLOCK_K == 0);

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;

    const size_t bytes_A = elements_A * sizeof(float);
    const size_t bytes_B = elements_B * sizeof(float);
    const size_t bytes_C = elements_C * sizeof(float);

    std::vector<float> h_A(elements_A);
    std::vector<float> h_B(elements_B);
    std::vector<float> h_C_custom(elements_C);
    std::vector<float> h_C_cublas(elements_C);

    srand(42);

    for (size_t i = 0; i < elements_A; ++i) {
        h_A[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }

    for (size_t i = 0; i < elements_B; ++i) {
        h_B[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_custom = nullptr;
    float* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_custom, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytes_C));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C_custom, 0, bytes_C));
    CUDA_CHECK(cudaMemset(d_C_cublas, 0, bytes_C));

    dim3 block_size(128);
    dim3 grid_size(
        (N + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    // ---------------------------------------------------------
    // 1. 创建 cuBLAS
    // ---------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    /*
     * 严格 FP32 模式，避免 TF32 Tensor Core 让对比差距过大。
     *
     * 如果想对比 cuBLAS 默认最快模式，可以注释这一行，
     * 或改成 CUBLAS_DEFAULT_MATH。
     */
    CUBLAS_CHECK(
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
     * cuBLAS 使用列主序，而这里的数据是行主序：
     *
     * C = A × B
     * C^T = B^T × A^T
     *
     * 因此调用 cuBLAS 时交换 A 和 B：
     *
     * m = N
     * n = M
     * k = K
     * cuBLAS A = 行主序 B
     * cuBLAS B = 行主序 A
     */
    auto launch_cublas = [&]() {
        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,                  // m
                M,                  // n
                K,                  // k
                &alpha,
                d_B, N,             // B 按列主序看作 B^T
                d_A, K,             // A 按列主序看作 A^T
                &beta,
                d_C_cublas, N
            )
        );
    };

    auto launch_custom = [&]() {
        gemm<
            M, N, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            TM, TN
        ><<<grid_size, block_size>>>(
            d_A,
            d_B,
            d_C_custom
        );
    };

    // ---------------------------------------------------------
    // 2. 分别预热
    // ---------------------------------------------------------
    for (int i = 0; i < WARMUP; ++i) {
        launch_custom();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < WARMUP; ++i) {
        launch_cublas();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------------------------------------------------
    // 3. 创建 CUDA Event
    // ---------------------------------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------------------------------------------------------
    // 4. 自定义 GEMM 计时
    // ---------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i) {
        launch_custom();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float custom_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &custom_total_ms, start, stop));

    const float custom_ms = custom_total_ms / REPEAT;

    // ---------------------------------------------------------
    // 5. cuBLAS 计时
    // ---------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i) {
        launch_cublas();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &cublas_total_ms, start, stop));

    const float cublas_ms = cublas_total_ms / REPEAT;

    // GEMM 浮点运算量约为 2*M*N*K
    const double operations =
        2.0 * static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    const double custom_tflops =
        operations / (custom_ms * 1.0e-3) / 1.0e12;

    const double cublas_tflops =
        operations / (cublas_ms * 1.0e-3) / 1.0e12;

    // ---------------------------------------------------------
    // 6. 拷贝结果并验证
    // ---------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_C_custom.data(),
        d_C_custom,
        bytes_C,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_C_cublas.data(),
        d_C_cublas,
        bytes_C,
        cudaMemcpyDeviceToHost
    ));

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double sum_abs_error = 0.0;
    size_t error_count = 0;

    constexpr double atol = 1e-2;
    constexpr double rtol = 1e-3;

    for (size_t i = 0; i < elements_C; ++i) {
        const double custom_value = h_C_custom[i];
        const double reference_value = h_C_cublas[i];

        const double abs_error =
            std::abs(custom_value - reference_value);

        const double rel_error =
            abs_error / (std::abs(reference_value) + 1e-6);

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        sum_abs_error += abs_error;

        if (abs_error > atol + rtol * std::abs(reference_value)) {
            ++error_count;
        }
    }

    const double mean_abs_error =
        sum_abs_error / static_cast<double>(elements_C);

    const double performance_ratio =
        custom_tflops / cublas_tflops * 100.0;

    printf("\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Warmup: %d, Repeat: %d\n", WARMUP, REPEAT);
    printf("\n");

    printf("Custom GEMM: %8.4f ms, %8.3f TFLOPS\n",
           custom_ms, custom_tflops);

    printf("cuBLAS SGEMM: %8.4f ms, %8.3f TFLOPS\n",
           cublas_ms, cublas_tflops);

    printf("Custom/cuBLAS: %8.2f%%\n", performance_ratio);
    printf("cuBLAS speedup: %8.3fx\n",
           custom_ms / cublas_ms);

    printf("\nCorrectness:\n");
    printf("Max absolute error: %.8e\n", max_abs_error);
    printf("Mean absolute error: %.8e\n", mean_abs_error);
    printf("Max relative error: %.8e\n", max_rel_error);
    printf("Mismatch count: %zu / %zu\n",
           error_count, elements_C);

    printf("Result: %s\n",
           error_count == 0 ? "PASS" : "FAIL");

    // ---------------------------------------------------------
    // 7. 清理
    // ---------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return 0;
}