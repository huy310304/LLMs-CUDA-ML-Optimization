#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Tile size for matrix multiplication

// Error checking macro for CUDA functions
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Matrix multiplication kernel (assuming square matrices)
__global__ void matrixMul(const float *A, const float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < size; ++k) {
        sum += A[row * size + k] * B[k * size + col];
    }
    C[row * size + col] = sum;
}

// Host function for matrix multiplication
void matrixMultiplication(const float *h_A, const float *h_B, float *h_C, int size) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Verification function: performs matrix multiplication on the CPU
void verifyMatrixMultiplication(const float *A, const float *B, float *C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    const int size = 1024;  // Size of matrices (assuming square matrices)
    float *h_A = (float *)malloc(size * size * sizeof(float));
    float *h_B = (float *)malloc(size * size * sizeof(float));
    float *h_C = (float *)malloc(size * size * sizeof(float));
    float *h_C_CPU = (float *)malloc(size * size * sizeof(float));  // For CPU result

    // Initialize matrices A and B with random values
    for (int i = 0; i < size * size; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Perform matrix multiplication on the CPU for verification
    verifyMatrixMultiplication(h_A, h_B, h_C_CPU, size);

    // Perform matrix multiplication on GPU
    matrixMultiplication(h_A, h_B, h_C, size);

    // Verify results by comparing GPU and CPU results
    float maxDiff = 0.0f;
    for (int i = 0; i < size * size; ++i) {
        float diff = fabs(h_C[i] - h_C_CPU[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }

    // Print maximum difference (should be close to zero for correct implementation)
    printf("Maximum difference between CPU and GPU results: %f\n", maxDiff);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}
