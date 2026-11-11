// PyTorch bindings: validate tensors, launch on the current stream, check for errors.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

void launch_vector_add(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream);
void launch_softmax(const float* in, float* out, int64_t n_rows, int n_cols, cudaStream_t stream);
#ifdef WITH_CUTLASS
int launch_cutlass_gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream);
#endif

#define CHECK_INPUT(t, dt)                                                   \
    TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor");                 \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");              \
    TORCH_CHECK((t).scalar_type() == (dt), #t " has the wrong dtype")

torch::Tensor vector_add(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x, torch::kFloat32);
    CHECK_INPUT(y, torch::kFloat32);
    TORCH_CHECK(x.numel() == y.numel(), "size mismatch");
    auto out = torch::empty_like(x);
    launch_vector_add(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), x.numel(),
                      at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor softmax(torch::Tensor x) {
    CHECK_INPUT(x, torch::kFloat32);
    TORCH_CHECK(x.dim() == 2, "softmax expects a 2-D tensor");
    auto out = torch::empty_like(x);
    launch_softmax(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), (int)x.size(1),
                   at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

#ifdef WITH_CUTLASS
torch::Tensor cutlass_gemm(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a, torch::kFloat16);
    CHECK_INPUT(b, torch::kFloat16);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "shape mismatch");
    const int m = (int)a.size(0), k = (int)a.size(1), n = (int)b.size(1);
    auto c = torch::empty({m, n}, a.options());
    int status = launch_cutlass_gemm(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k,
                                     at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == 0, "CUTLASS GEMM failed with status ", status,
                " (M, N, K must be multiples of 8; requires SM80+)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "x + y (fp32)");
    m.def("softmax", &softmax, "row-wise softmax (fp32, 2-D)");
#ifdef WITH_CUTLASS
    m.def("cutlass_gemm", &cutlass_gemm, "A @ B (fp16, row-major, CUTLASS)");
#endif
}
