#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>

__constant__ float c_Lx, c_Ly, c_Lz;
__constant__ float c_h, c_inv_h2;
__constant__ float c_tau, c_tau2, c_a2, c_Laplacian;
__constant__ int c_N, c_M, c_K;
__constant__ int c_NLow, c_MLow, c_KLow;
__constant__ int c_NGlobal, c_MGlobal, c_KGlobal;
__constant__ int c_MLow0, c_JRight, c_MK;

#define PI 3.14159265359
#define DIMENTIONS 3

__device__ float analyticalSolution_device(float x, float y, float z, float t) {
    float at = PI/3.0f * sqrtf(4.0f/(c_Lx*c_Lx) + 1.0f/(c_Ly*c_Ly) + 4.0f/(c_Lz*c_Lz));
    return sinf(2.0f*PI*x/c_Lx) * sinf((1.0f + y/c_Ly) * PI) * 
           sinf((1.0f + z/c_Lz) * 2.0f*PI) * cosf(at*t + PI);
}

__device__ float phi_device(float x, float y, float z) {
    return analyticalSolution_device(x, y, z, 0.0f);
}

__global__ void initialize_u0_kernel(float* res) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_N || j >= c_M || k >= c_K) return;
    
    int idx = i *c_MK + j * c_K + k;

    if ((c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) {
        res[idx] = 0.0f;
        return;
    }

    float x = (i + c_NLow) * c_h;
    float y = (j + c_MLow) * c_h;
    float z = (k + c_KLow) * c_h;

    res[idx] = phi_device(x, y, z);
}

__global__ void initialize_u1_kernel(float* res, float* ut,
                                     float* xPrev, float* xNext, float* yPrev, float* yNext,
                                     float* zPrev, float* zNext) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= c_N || j >= c_M || k >= c_K) return;
    int idx = i * c_MK + j * c_K + k;
    if ((c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) {
        res[idx] = 0.0f;
        return;
    }
    float center = ut[idx];

    float left = (i > 0) ? ut[idx - c_MK] : xPrev[j*c_K + k];
    float right = (i < c_N-1) ? ut[idx + c_MK] : xNext[j*c_K + k];
    float front = (j > 0) ? ut[idx - c_K] : yPrev[i*c_K + k];
    float back = (j < c_M-1) ? ut[idx + c_K] : yNext[i*c_K + k];
    float bottom = (k > 0) ? ut[idx - 1] : zPrev[i*c_M + j];
    float top = (k < c_K-1) ? ut[idx + 1] : zNext[i*c_M + j];

    float laplacian = (left + right + front + back + bottom + top - 6.0f * center);

    float l = c_Laplacian * laplacian / 2.0f;
    res[idx] = center + l;
}

void makeU0_device(float* d_ut0, dim3 blockSize, dim3 gridSize) {
    initialize_u0_kernel<<<gridSize, blockSize>>>(d_ut0);
    cudaDeviceSynchronize();
}

void makeU1_device(float* d_ut1, float* d_ut0,
                     dim3 blockSize, dim3 gridSize,
                     float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                     float* d_zPrev, float* d_zNext) {
    initialize_u1_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0,
                                                     d_xPrev, d_xNext, d_yPrev, d_yNext,
                                                     d_zPrev, d_zNext);
    cudaDeviceSynchronize();
}

__global__ void step_kernel(float* ut1, float* ut0,
                            float* xPrev, float* xNext, 
                            float* yPrev, float* yNext,
                            float* zPrev, float* zNext) {

    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= c_N) || (j >= c_M) || (k >= c_K) || (c_MLow0 && j == 0) || (c_JRight && j == c_M - 1)) return;
    
    int idx = i * c_MK + j * c_K + k;

    float centerx2 = 2.0f * ut1[idx];

    float left = (i > 0) ? ut1[idx - c_MK] : xPrev[j*c_K + k];
    float right = (i < c_N-1) ? ut1[idx + c_MK] : xNext[j*c_K + k];
    float front = (j > 0) ? ut1[idx - c_K] : yPrev[i*c_K + k];
    float back = (j < c_M-1) ? ut1[idx + c_K] : yNext[i*c_K + k];
    float bottom = (k > 0) ? ut1[idx - 1] : zPrev[i*c_M + j];
    float top = (k < c_K-1) ? ut1[idx + 1] : zNext[i*c_M + j];

    float laplacian = (left + right + front + back + bottom + top - 3.0f * centerx2);

    float l = c_Laplacian * laplacian;

    float dt2 = centerx2 - ut0[idx];
    ut0[idx] = dt2 + l;
}

void makeStep_device(float* d_ut1, float* d_ut0,
                     dim3 blockSize, dim3 gridSize,
                     float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                     float* d_zPrev, float* d_zNext) {
    step_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0,
                                            d_xPrev, d_xNext, d_yPrev, d_yNext,
                                            d_zPrev, d_zNext);
}

void freeBuffersAndTerminate(float** deviceBuffers, int numDeviceBuffers) {
    for (int i = 0; i < numDeviceBuffers; i++)
        if (deviceBuffers[i] != NULL)
            cudaFree(deviceBuffers[i]);
}

void setup_constants(int N, int M, int K, int NLow, int MLow, int KLow,
                     int NGlobal, int MGlobal, int KGlobal,
                     float Lx, float Ly, float Lz, 
                     float h, float tau, float a2) {

    float inv_h2 = 1.0f / (h * h);
    float tau2 = tau * tau;
    float laplacian = a2 * inv_h2 * tau2;

    int JRight = (M + MLow >= MGlobal), MLow0 = (MLow == 0), MK = M*K;

    cudaMemcpyToSymbol(c_JRight, &JRight, sizeof(int));
    cudaMemcpyToSymbol(c_MLow0, &MLow0, sizeof(int));
    cudaMemcpyToSymbol(c_MK, &MK, sizeof(int));

    cudaMemcpyToSymbol(c_Lx, &Lx, sizeof(float));
    cudaMemcpyToSymbol(c_Ly, &Ly, sizeof(float));
    cudaMemcpyToSymbol(c_Lz, &Lz, sizeof(float));
    
    cudaMemcpyToSymbol(c_Laplacian, &laplacian, sizeof(float));
    cudaMemcpyToSymbol(c_h, &h, sizeof(float));
    cudaMemcpyToSymbol(c_inv_h2, &inv_h2, sizeof(float));
    
    cudaMemcpyToSymbol(c_tau, &tau, sizeof(float));
    cudaMemcpyToSymbol(c_tau2, &tau2, sizeof(float));
    cudaMemcpyToSymbol(c_a2, &a2, sizeof(float));
    
    cudaMemcpyToSymbol(c_N, &N, sizeof(int));
    cudaMemcpyToSymbol(c_M, &M, sizeof(int));
    cudaMemcpyToSymbol(c_K, &K, sizeof(int));
    
    cudaMemcpyToSymbol(c_NLow, &NLow, sizeof(int));
    cudaMemcpyToSymbol(c_MLow, &MLow, sizeof(int));
    cudaMemcpyToSymbol(c_KLow, &KLow, sizeof(int));
    
    cudaMemcpyToSymbol(c_NGlobal, &NGlobal, sizeof(int));
    cudaMemcpyToSymbol(c_MGlobal, &MGlobal, sizeof(int));
    cudaMemcpyToSymbol(c_KGlobal, &KGlobal, sizeof(int));
}

int main(int argc, char *argv[]) {

    int NGlobal = 512;
    float T = 1.0, L = 1.0;
    float h = L/(NGlobal-1), a2 = 1.0/9.0, tau = h / (sqrt(a2 * 12));

    int N = NGlobal;
    int M = NGlobal;
    int K = NGlobal;

    int NLow = 0;
    int MLow = 0;
    int KLow = 0;

    int uShape = N*M*K;
    setup_constants(N, M, K, NLow, MLow, KLow,
                     NGlobal, NGlobal, NGlobal,
                     L, L, L, h, tau, a2);

    float *d_ut0 = NULL, *d_ut1 = NULL, *d_xPrev = NULL, *d_xNext = NULL, *d_yPrev = NULL, *d_yNext = NULL, *d_zPrev = NULL, *d_zNext = NULL;
    cudaMalloc(&d_ut0, uShape * sizeof(float)); cudaMalloc(&d_ut1, uShape * sizeof(float));
    cudaMalloc(&d_xPrev, M*K * sizeof(float)); cudaMalloc(&d_xNext, M*K * sizeof(float)); cudaMalloc(&d_yPrev, N*K * sizeof(float)); cudaMalloc(&d_yNext, N*K * sizeof(float));
    cudaMalloc(&d_zPrev, N*M * sizeof(float)); cudaMalloc(&d_zNext, N*M * sizeof(float));

    cudaMemset(d_xPrev, 0, M*K * sizeof(float)); cudaMemset(d_xNext, 0, M*K * sizeof(float));
    cudaMemset(d_yPrev, 0, N*K * sizeof(float)); cudaMemset(d_yNext, 0, N*K * sizeof(float));
    cudaMemset(d_zPrev, 0, N*M * sizeof(float)); cudaMemset(d_zNext, 0, N*M * sizeof(float));

    float* deviceBuffers[9] = {
        d_ut0, d_ut1, d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext
    };

    cudaSetDevice(0);

    dim3 blockSize(64, 4, 2);
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        (K + blockSize.z - 1) / blockSize.z
    );

    makeU0_device(d_ut0, blockSize, gridSize);

    makeU1_device(d_ut1, d_ut0, blockSize, gridSize,
                  d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext);


    for (int t = 0; t * tau < T; t++) {
        makeStep_device(d_ut1, d_ut0, blockSize, gridSize, d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext);
        cudaDeviceSynchronize();
    }

    freeBuffersAndTerminate(deviceBuffers, 9);
    return 0;
}


// mpirun -np 1 nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency,shared_utilization,achieved_occupancy,inst_replay_overhead,branch_efficiency,dram_utilization ./step_nvprof.out