// Exact-order CUDA implementation of scipy.ndimage.uniform_filter1d.
//
// Each CUDA thread processes one complete image line.  This deliberately
// preserves SciPy 1.10.1's double-precision initial sum and sliding update
// order instead of using a parallel prefix sum, whose rounding differs.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ int64_t reflect_index(int64_t position,
                                                 int64_t length) {
    const int64_t period = 2 * length;
    int64_t folded = position % period;
    if (folded < 0) {
        folded += period;
    }
    return folded < length ? folded : period - 1 - folded;
}

__global__ void uniform_axis0(const float* input,
                              float* output,
                              int64_t batch,
                              int64_t height,
                              int64_t width,
                              int kernel_size) {
    const int64_t line =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (line >= batch * width) {
        return;
    }
    const int64_t batch_index = line / width;
    const int64_t column = line % width;
    const int64_t image_base = batch_index * height * width;
    const int64_t pad = kernel_size / 2;
    double accumulator = 0.0;
    for (int offset = 0; offset < kernel_size; ++offset) {
        const int64_t row = reflect_index(offset - pad, height);
        accumulator += static_cast<double>(
            input[image_base + row * width + column]
        );
    }
    output[image_base + column] = static_cast<float>(
        accumulator / static_cast<double>(kernel_size)
    );
    for (int64_t row = 1; row < height; ++row) {
        const int64_t incoming = reflect_index(row + pad, height);
        const int64_t outgoing = reflect_index(row - pad - 1, height);
        accumulator += static_cast<double>(
                           input[image_base + incoming * width + column]
                       )
                     - static_cast<double>(
                           input[image_base + outgoing * width + column]
                       );
        output[image_base + row * width + column] = static_cast<float>(
            accumulator / static_cast<double>(kernel_size)
        );
    }
}

__global__ void uniform_axis1(const float* input,
                              float* output,
                              int64_t batch,
                              int64_t height,
                              int64_t width,
                              int kernel_size) {
    const int64_t line =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (line >= batch * height) {
        return;
    }
    const int64_t batch_index = line / height;
    const int64_t row = line % height;
    const int64_t pad = kernel_size / 2;
    const int64_t base = batch_index * height * width + row * width;
    double accumulator = 0.0;
    for (int offset = 0; offset < kernel_size; ++offset) {
        const int64_t column = reflect_index(offset - pad, width);
        accumulator += static_cast<double>(input[base + column]);
    }
    output[base] = static_cast<float>(
        accumulator / static_cast<double>(kernel_size)
    );
    for (int64_t column = 1; column < width; ++column) {
        const int64_t incoming = reflect_index(column + pad, width);
        const int64_t outgoing = reflect_index(column - pad - 1, width);
        accumulator += static_cast<double>(input[base + incoming])
                     - static_cast<double>(input[base + outgoing]);
        output[base + column] = static_cast<float>(
            accumulator / static_cast<double>(kernel_size)
        );
    }
}

}  // namespace

extern "C" int launch_uniform_filter_axis(
    const float* input,
    float* output,
    int64_t batch,
    int64_t height,
    int64_t width,
    int axis,
    int kernel_size,
    uint64_t stream_address
) {
    if (input == nullptr || output == nullptr || batch <= 0 || height <= 0 ||
        width <= 0 || kernel_size <= 0 || (axis != 0 && axis != 1)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_address);
    constexpr int threads = 256;
    if (axis == 0) {
        const int64_t line_count = batch * width;
        const int blocks = static_cast<int>(
            (line_count + threads - 1) / threads
        );
        uniform_axis0<<<blocks, threads, 0, stream>>>(
            input, output, batch, height, width, kernel_size
        );
    } else {
        const int64_t line_count = batch * height;
        const int blocks = static_cast<int>(
            (line_count + threads - 1) / threads
        );
        uniform_axis1<<<blocks, threads, 0, stream>>>(
            input, output, batch, height, width, kernel_size
        );
    }
    return static_cast<int>(cudaGetLastError());
}
