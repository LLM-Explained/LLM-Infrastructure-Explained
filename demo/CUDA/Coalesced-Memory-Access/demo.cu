#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int BLOCK_SIZE = 256;

__global__ void coalesced_copy_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

__global__ void misaligned_copy_kernel(const float* x, float* y, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src = idx + offset;
    if (idx < n - offset) {
        y[idx] = x[src];
    }
}

__global__ void strided_copy_kernel(const float* x, float* y, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int src = tid * stride;
    if (src < n) {
        y[tid] = x[src];
    }
}

float run_kernel_ms(void (*launch_fn)(const float*, float*, int, cudaStream_t),
                    const float* d_x, float* d_y, int n, int iters = 100) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_fn(d_x, d_y, n, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

void launch_coalesced(const float* d_x, float* d_y, int n, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    coalesced_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_x, d_y, n);
}

struct MisalignedLauncher {
    int offset;
    static MisalignedLauncher* self;
    static void launch(const float* d_x, float* d_y, int n, cudaStream_t stream) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        misaligned_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_x, d_y, n, self->offset);
    }
};
MisalignedLauncher* MisalignedLauncher::self = nullptr;

struct StridedLauncher {
    int stride;
    static StridedLauncher* self;
    static void launch(const float* d_x, float* d_y, int n, cudaStream_t stream) {
        int out_n = n / self->stride;
        int blocks = (out_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        strided_copy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_x, d_y, n, self->stride);
    }
};
StridedLauncher* StridedLauncher::self = nullptr;

int main() {
    const int n = 1 << 24;  // ~16M floats
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    std::iota(h_x.begin(), h_x.end(), 0.0f);

    float *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    float ms_coalesced = run_kernel_ms(launch_coalesced, d_x, d_y, n);

    MisalignedLauncher misaligned{1};
    MisalignedLauncher::self = &misaligned;
    float ms_misaligned = run_kernel_ms(MisalignedLauncher::launch, d_x, d_y, n);

    StridedLauncher strided{8};
    StridedLauncher::self = &strided;
    float ms_strided = run_kernel_ms(StridedLauncher::launch, d_x, d_y, n);

    std::cout << "=== Coalesced global memory access demo ===\n\n";
    std::cout << "Array size: " << n << " floats\n\n";
    std::cout << "Average kernel time per launch:\n";
    std::cout << "  coalesced copy : " << ms_coalesced << " ms\n";
    std::cout << "  misaligned copy: " << ms_misaligned << " ms\n";
    std::cout << "  strided copy   : " << ms_strided << " ms\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "- Coalesced copy uses thread k -> word k.\n";
    std::cout << "- Misaligned copy shifts the starting address and can force extra segments.\n";
    std::cout << "- Strided copy spreads warp accesses apart and usually hurts coalescing badly.\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
}
