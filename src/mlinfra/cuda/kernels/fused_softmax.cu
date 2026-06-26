// Numerically-stable row-wise softmax, one thread block per row.
//
// This is the softmax that sits inside attention: subtract the row max before exp() to avoid
// overflow, reduce in shared memory, then normalize. A "fused" kernel like this avoids the
// extra global-memory round-trips a naive max -> exp -> sum -> div sequence would incur.

extern "C" __global__
void softmax_rows(const float* in, float* out, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = in + row * cols;
    float* y = out + row * cols;

    // 1) row max (parallel reduction over a strided slice per thread)
    float local_max = -3.402823466e+38f;  // -FLT_MAX
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, x[j]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // 2) exp(x - max) and accumulate the denominator
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = __expf(x[j] - row_max);
        y[j] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // 3) normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y[j] *= inv_sum;
    }
}
