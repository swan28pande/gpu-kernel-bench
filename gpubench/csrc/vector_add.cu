#include <cstdint>
#include <cuda_runtime.h>

// Grid-stride loop, float4-vectorised when the pointers allow it.
__global__ void vector_add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                  float* __restrict__ out, int64_t n) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) out[i] = x[i] + y[i];
}

__global__ void vector_add_kernel_vec4(const float4* __restrict__ x, const float4* __restrict__ y,
                                       float4* __restrict__ out, int64_t n4) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 a = x[i], b = y[i];
        out[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

void launch_vector_add(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    bool aligned = ((uintptr_t)x | (uintptr_t)y | (uintptr_t)out) % 16 == 0;
    if (aligned && n % 4 == 0) {
        int64_t n4 = n / 4;
        int blocks = (int)((n4 + threads - 1) / threads < 65535 ? (n4 + threads - 1) / threads : 65535);
        vector_add_kernel_vec4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(y),
            reinterpret_cast<float4*>(out), n4);
    } else {
        int blocks = (int)((n + threads - 1) / threads < 65535 ? (n + threads - 1) / threads : 65535);
        vector_add_kernel<<<blocks, threads, 0, stream>>>(x, y, out, n);
    }
}
