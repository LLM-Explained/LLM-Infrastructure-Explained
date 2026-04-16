#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// ------------------------------------------------------------
// Kernel 1: transpose-style shared-memory tile WITHOUT padding
// This is the classic bank-conflict pattern.
// ------------------------------------------------------------
__global__ void transpose_no_padding(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // coalesced global load into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    // transpose-like shared-memory read
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ------------------------------------------------------------
// Kernel 2: same pattern WITH padding
// The +1 stride breaks the harmful bank alignment.
// ------------------------------------------------------------
__global__ void transpose_with_padding(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

float run_kernel(void (*kernel)(const float*, float*, int, int),
                 const float* d_in, float* d_out,
                 int width, int height,
                 int iters = 100) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

int main() {
    const int width = 4096;
    const int height = 4096;
    const size_t numel = static_cast<size_t>(width) * height;
    const size_t bytes = numel * sizeof(float);

    std::vector<float> h_in(numel, 1.0f);
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    float ms_no_pad = run_kernel(transpose_no_padding, d_in, d_out, width, height);
    float ms_pad = run_kernel(transpose_with_padding, d_in, d_out, width, height);

    std::cout << "=== Shared-memory tiling + bank-conflict avoidance demo ===\n\n";
    std::cout << "Matrix size: " << width << " x " << height << "\n";
    std::cout << "Tile size  : " << TILE_DIM << " x " << TILE_DIM << "\n\n";

    std::cout << "Average kernel time per launch:\n";
    std::cout << "  no padding   : " << ms_no_pad << " ms\n";
    std::cout << "  with padding : " << ms_pad << " ms\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "- Both kernels use shared-memory tiling to improve global-memory reuse.\n";
    std::cout << "- The unpadded tile creates a classic transpose-style shared-memory bank-conflict pattern.\n";
    std::cout << "- The padded tile changes the stride and avoids the worst bank conflicts.\n";
    std::cout << "- This is the core backbone of tiling + bank-conflict avoidance.\n";

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
