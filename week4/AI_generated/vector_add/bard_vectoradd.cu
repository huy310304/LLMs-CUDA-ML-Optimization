#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float* d_a, float* d_b, float* d_c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    d_c[i] = d_a[i] + d_b[i];
  }
}

int main() {
  // Define vector size
  int N = 1024*1024;

  // Allocate memory on host for the vectors
  float* h_a = (float*)malloc(N * sizeof(float));
  float* h_b = (float*)malloc(N * sizeof(float));
  float* h_c = (float*)malloc(N * sizeof(float));

  // Initialize the vectors on host (example)
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // Allocate memory on device for the vectors
  float* d_a;
  float* d_b;
  float* d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  // Define number of threads per block and number of blocks
  int threadsPerBlock = 256;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel
  vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Copy data back from device to host
  cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free memory on host
  free(h_a);
  free(h_b);
  free(h_c);

  // (Optional) Verify results on host

  return 0;
}