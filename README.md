# CUDA Kernel Optimization and Implementation

* Implemented core CUDA C++ kernels for LLM inference, including Reduce, GEMM, Fused Residual Add + RMSNorm, SiLU × Mul, Attention, and FlashAttention; validated numerical correctness and benchmarked performance against PyTorch and cuBLAS baselines.
* Optimized frequently used kernels such as GEMM and RMSNorm using shared-memory tiling, register tiling, warp-level parallelism, and vectorized memory access to improve data reuse and memory-access efficiency.
* Integrated the custom kernels with PyTorch through the C++/CUDA Extension mechanism, using `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` for operator registration and Python invocation.
* Achieved approximately 21.77 TFLOPS for FP32 GEMM with \(M=N=K=4096\), reaching 90.3% of cuBLAS performance under the same configuration.
* Implemented a simplified FlashAttention forward kernel using block tiling and online softmax for `seq_len=1024` and `head_dim=128`; reduced intermediate-memory usage by approximately 70% and achieved a 2.5× end-to-end speedup over naive Attention.
