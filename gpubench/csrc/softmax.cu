#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// Row-wise softmax: one block per row, three passes (max, sum, normalise).

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <bool IS_MAX>
__device__ float block_reduce(float v, float* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    const float identity = IS_MAX ? -INFINITY : 0.f;
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = threadIdx.x < n_warps ? smem[threadIdx.x] : identity;
    if (warp == 0) v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();  // smem is reused by the next reduction
    return v;
}

__global__ void softmax_kernel(const float* __restrict__ in, float* __restrict__ out, int n_cols) {
    __shared__ float smem[32];
    const float* row = in + (int64_t)blockIdx.x * n_cols;
    float* orow = out + (int64_t)blockIdx.x * n_cols;

    float m = -INFINITY;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) m = fmaxf(m, row[c]);
    m = block_reduce<true>(m, smem);

    float s = 0.f;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) s += __expf(row[c] - m);
    s = block_reduce<false>(s, smem);

    const float inv = 1.f / s;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) orow[c] = __expf(row[c] - m) * inv;
}

void launch_softmax(const float* in, float* out, int64_t n_rows, int n_cols, cudaStream_t stream) {
    const int threads = n_cols >= 4096 ? 1024 : n_cols >= 1024 ? 512 : 256;
    softmax_kernel<<<(unsigned)n_rows, threads, 0, stream>>>(in, out, n_cols);
}
