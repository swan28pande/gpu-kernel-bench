// fp16 GEMM via CUTLASS 2.x device API: C = A @ B, row-major, fp32 accumulate,
// tensor cores on SM80+. Requires M, N, K to be multiples of 8 (128-bit alignment).
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

using RowMajor = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, RowMajor,   // A
    cutlass::half_t, RowMajor,   // B
    cutlass::half_t, RowMajor,   // C / D
    float,                       // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80>;

// Returns 0 on success, otherwise the cutlass::Status code.
int launch_cutlass_gemm(const void* a, const void* b, void* c, int m, int n, int k,
                        cudaStream_t stream) {
    using T = cutlass::half_t;
    Gemm::Arguments args({m, n, k},
                         {static_cast<const T*>(a), k},
                         {static_cast<const T*>(b), n},
                         {static_cast<T*>(c), n},
                         {static_cast<T*>(c), n},
                         {1.0f, 0.0f});
    Gemm gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status == cutlass::Status::kSuccess) status = gemm(args, nullptr, stream);
    return static_cast<int>(status);
}
