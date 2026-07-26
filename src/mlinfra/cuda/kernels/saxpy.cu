// SAXPY: out = a * x + y. The "hello world" of CUDA — used as a compile smoke test.
extern "C" __global__
void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}
