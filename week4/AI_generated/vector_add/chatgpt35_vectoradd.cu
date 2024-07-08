#include <stdio.h>

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread index is within bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Size of vectors
    int n = 1024*1024; // for example, adjust this as needed
    
    // Host vectors
    int *h_a, *h_b, *h_c;
    
    // Device vectors
    int *d_a, *d_b, *d_c;
    
    // Allocate memory on host
    h_a = (int *)malloc(n * sizeof(int));
    h_b = (int *)malloc(n * sizeof(int));
    h_c = (int *)malloc(n * sizeof(int));
    
    // Allocate memory on device
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMalloc((void **)&d_b, n * sizeof(int));
    cudaMalloc((void **)&d_c, n * sizeof(int));
    
    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result from device to host
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error: Mismatch at index %d\n", i);
            break;
        }
    }
    
    printf("Vector addition completed successfully!\n");
    
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}