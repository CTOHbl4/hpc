#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>
#include <mpi.h>

#define PI 3.14159265359
#define DIMENTIONS 3

__device__ float analyticalSolution_device(float Lx, float Ly, float Lz, 
                                          float x, float y, float z, float t) {
    float at = PI/3.0f * sqrtf(4.0f/(Lx*Lx) + 1.0f/(Ly*Ly) + 4.0f/(Lz*Lz));
    return sinf(2.0f*PI*x/Lx) * sinf((1.0f + y/Ly) * PI) * 
           sinf((1.0f + z/Lz) * 2.0f*PI) * cosf(at*t + PI);
}

__device__ float phi_device(float Lx, float Ly, float Lz, 
                           float x, float y, float z) {
    return analyticalSolution_device(Lx, Ly, Lz, x, y, z, 0.0f);
}

__device__ float aaLaplasian_cuda(float* ut, int N, int M, int K, int i, int j, int k, float h, float a2,
                                  float* xPrev, float* xNext, float* yPrev, float* yNext,
                                  float* zPrev, float* zNext, int MLow, int MGlobal) {
    int mk = M * K;
    int jk = j * K;
    int imk = mk * i;
    int idx = imk + jk + k;

    float left_val, right_val, front_val, back_val, bottom_val, top_val;
    float center_val = ut[idx];

    if (i == 0) {
        left_val = xPrev[jk + k];
    } else {
        left_val = ut[idx - mk];
    }
    
    if (i == N - 1) {
        right_val = xNext[jk + k];
    } else {
        right_val = ut[idx + mk];
    }

    if (j == 0) {
        if (MLow == 0) {
            front_val = 0.0f;
        } else {
            front_val = yPrev[i * K + k];
        }
    } else {
        front_val = ut[idx - K];
    }

    if (j == M - 1) {
        if (MLow + M >= MGlobal) {
            back_val = 0.0f;
        } else {
            back_val = yNext[i * K + k];
        }
    } else {
        back_val = ut[idx + K];
    }

    if (k == 0) {
        bottom_val = zPrev[i * M + j];
    } else {
        bottom_val = ut[idx - 1];
    }

    if (k == K - 1) {
        top_val = zNext[i * M + j];
    } else {
        top_val = ut[idx + 1];
    }

    float d2x = left_val - 2 * center_val + right_val;
    float d2y = front_val - 2 * center_val + back_val;
    float d2z = bottom_val - 2 * center_val + top_val;

    return a2 * (d2x + d2y + d2z) / (h * h);
}

__global__ void initialize_u0_kernel(float* res, int NLow, int MLow, int KLow,
                                     int N, int M, int K, int NGlobal, int MGlobal, int KGlobal,
                                     float Lx, float Ly, float Lz, float h) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= M || k >= K) return;
    
    int idx = i * M * K + j * K + k;

    if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M - 1)) {
        res[idx] = 0.0f;
        return;
    }

    float x = (i + NLow) * h;
    float y = (j + MLow) * h;
    float z = (k + KLow) * h;

    res[idx] = phi_device(Lx, Ly, Lz, x, y, z);
}

__global__ void initialize_u1_kernel(float* res, float* ut0, int NLow, int MLow, int KLow,
                                     int N, int M, int K, int NGlobal, int MGlobal, int KGlobal,
                                     float Lx, float Ly, float Lz, float h, float tau, float a2,
                                     float* xPrev, float* xNext, float* yPrev, float* yNext,
                                     float* zPrev, float* zNext) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= M || k >= K) return;
    int idx = i * M * K + j * K + k;
    if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M - 1)) {
        res[idx] = 0.0f;
        return;
    }
    float l = tau * tau * aaLaplasian_cuda(ut0, N, M, K, i, j, k, h, a2,
                                          xPrev, xNext, yPrev, yNext,
                                          zPrev, zNext, MLow, MGlobal) / 2.0f;
    res[idx] = ut0[idx] + l;
}

void makeU0_device(float* d_ut0, int NLow, int MLow, int KLow, int N, int M, int K,
                     int NGlobal, int MGlobal, int KGlobal,
                     float Lx, float Ly, float Lz, float h) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        (K + blockSize.z - 1) / blockSize.z
    );

    initialize_u0_kernel<<<gridSize, blockSize>>>(d_ut0, NLow, MLow, KLow,
                                                 N, M, K, NGlobal, MGlobal, KGlobal,
                                                 Lx, Ly, Lz, h);
    cudaDeviceSynchronize();
}

void makeU1_device(float* d_ut1, float* d_ut0, int NLow, int MLow, int KLow,
                     int N, int M, int K, int NGlobal, int MGlobal, int KGlobal,
                     float Lx, float Ly, float Lz, float h, float tau, float a2,
                     float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                     float* d_zPrev, float* d_zNext) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        (K + blockSize.z - 1) / blockSize.z
    );
    
    initialize_u1_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0, NLow, MLow, KLow,
                                                     N, M, K, NGlobal, MGlobal, KGlobal,
                                                     Lx, Ly, Lz, h, tau, a2,
                                                     d_xPrev, d_xNext, d_yPrev, d_yNext,
                                                     d_zPrev, d_zNext);
    cudaDeviceSynchronize();
}

__global__ void step_kernel(float* ut1, float* ut0, int NLow, int MLow, int KLow, 
                            int N, int M, int K, int NGlobal, int MGlobal, int KGlobal,
                            float h, float tau, float a2,
                            float* xPrev, float* xNext, 
                            float* yPrev, float* yNext,
                            float* zPrev, float* zNext) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= M || k >= K) return;

    if ((MLow == 0 && j == 0) || (MLow + M >= MGlobal && j == M - 1)) return;
    
    int idx = i * M * K + j * K + k;

    float l = tau * tau * aaLaplasian_cuda(ut1, N, M, K, i, j, k, h, a2,
                                          xPrev, xNext, yPrev, yNext,
                                          zPrev, zNext, MLow, MGlobal);
    float dt2 = 2 * ut1[idx] - ut0[idx];
    ut0[idx] = dt2 + l;
}

__global__ void pack_z_boundary_kernel(float* local_data, float* z_send, 
                                       int N, int M, int K, int exchange_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= N || j >= M) return;
    
    int src_idx = i * M * K + j * K + exchange_idx;
    int dst_idx = i * M + j;
    
    z_send[dst_idx] = local_data[src_idx];
}

__global__ void pack_y_boundary_kernel(float* local_data, float* y_send,
                                       int N, int M, int K, int j_offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= N || k >= K) return;
    
    int src_idx = i * M * K + j_offset * K + k;
    int dst_idx = i * K + k;
    
    y_send[dst_idx] = local_data[src_idx];
}

void exchange_cuda(float* d_local_data, int N, int M, int K, MPI_Comm cart_comm,
                   float* h_xPrev, float* h_xNext, float* h_yPrev, float* h_yNext,
                   float* h_zPrev, float* h_zNext,
                   float* h_xPrevSend, float* h_xNextSend, float* h_yPrevSend, float* h_yNextSend,
                   float* h_zPrevSend, float* h_zNextSend,
                   float* d_xPrev, float* d_xNext, float* d_yPrev, float* d_yNext,
                   float* d_zPrev, float* d_zNext,
                   int left_rank, int right_rank, int front_rank, int back_rank,
                   int down_rank, int up_rank, int NLow, int NGlobal, int MLow, 
                   int MGlobal, int KLow, int KGlobal) {
    cudaDeviceSynchronize();
    
    MPI_Request requests[12];
    int count = 0;
    
    // x
    int exchangeIdxXPrev = 0;
    int exchangeIdxXNext = (N - 1) * M * K;
    if (NLow == 0) {
        exchangeIdxXPrev = M * K;
    }
    if (NLow + N >= NGlobal) {
        exchangeIdxXNext = (N - 2) * M * K;
    }

    cudaMemcpy(h_xPrevSend, d_local_data + exchangeIdxXPrev, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xNextSend, d_local_data + exchangeIdxXNext, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Isend(h_xPrevSend, M * K, MPI_FLOAT, left_rank, 0, cart_comm, &requests[count++]);
    MPI_Irecv(h_xPrev, M * K, MPI_FLOAT, left_rank, 1, cart_comm, &requests[count++]);

    MPI_Isend(h_xNextSend, M * K, MPI_FLOAT, right_rank, 1, cart_comm, &requests[count++]);
    MPI_Irecv(h_xNext, M * K, MPI_FLOAT, right_rank, 0, cart_comm, &requests[count++]);
    
    // z
    int exchangeIdxZPrev = 0;
    int exchangeIdxZNext = K - 1;
    if (KLow == 0) {
        exchangeIdxZPrev = 1;
    }
    if (KLow + K >= KGlobal) {
        exchangeIdxZNext = K - 2;
    }
    dim3 blockSize_z(16, 16);
    dim3 gridSize_z((N + blockSize_z.x - 1) / blockSize_z.x,
                    (M + blockSize_z.y - 1) / blockSize_z.y);

    pack_z_boundary_kernel<<<gridSize_z, blockSize_z>>>(
        d_local_data, d_zPrev, N, M, K, exchangeIdxZPrev);
    pack_z_boundary_kernel<<<gridSize_z, blockSize_z>>>(
        d_local_data, d_zNext, N, M, K, exchangeIdxZNext);

    cudaMemcpy(h_zPrevSend, d_zPrev, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zNextSend, d_zNext, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Isend(h_zPrevSend, N * M, MPI_FLOAT, down_rank, 4, cart_comm, &requests[count++]);
    MPI_Irecv(h_zPrev, N * M, MPI_FLOAT, down_rank, 5, cart_comm, &requests[count++]);
    
    MPI_Isend(h_zNextSend, N * M, MPI_FLOAT, up_rank, 5, cart_comm, &requests[count++]);
    MPI_Irecv(h_zNext, N * M, MPI_FLOAT, up_rank, 4, cart_comm, &requests[count++]);

    if (front_rank != MPI_PROC_NULL) {
        dim3 blockSize_y(16, 16);
        dim3 gridSize_y((N + blockSize_y.x - 1) / blockSize_y.x,
                        (K + blockSize_y.y - 1) / blockSize_y.y);

        pack_y_boundary_kernel<<<gridSize_y, blockSize_y>>>(
            d_local_data, d_yPrev, N, M, K, 0);

        cudaMemcpy(h_yPrevSend, d_yPrev, N * K * sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Isend(h_yPrevSend, N * K, MPI_FLOAT, front_rank, 2, cart_comm, &requests[count++]);
        MPI_Irecv(h_yPrev, N * K, MPI_FLOAT, front_rank, 3, cart_comm, &requests[count++]);
    }

    if (back_rank != MPI_PROC_NULL) {
        dim3 blockSize_y(16, 16);
        dim3 gridSize_y((N + blockSize_y.x - 1) / blockSize_y.x,
                        (K + blockSize_y.y - 1) / blockSize_y.y);

        pack_y_boundary_kernel<<<gridSize_y, blockSize_y>>>(
            d_local_data, d_yNext, N, M, K, M - 1);

        cudaMemcpy(h_yNextSend, d_yNext, N * K * sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Isend(h_yNextSend, N * K, MPI_FLOAT, back_rank, 3, cart_comm, &requests[count++]);
        MPI_Irecv(h_yNext, N * K, MPI_FLOAT, back_rank, 2, cart_comm, &requests[count++]);
    }

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

    cudaMemcpy(d_xPrev, h_xPrev, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xNext, h_xNext, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_zPrev, h_zPrev, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zNext, h_zNext, N * M * sizeof(float), cudaMemcpyHostToDevice);

    if (front_rank != MPI_PROC_NULL) {
        cudaMemcpy(d_yPrev, h_yPrev, N * K * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    if (back_rank != MPI_PROC_NULL) {
        cudaMemcpy(d_yNext, h_yNext, N * K * sizeof(float), cudaMemcpyHostToDevice);
    }
}

__global__ void compute_errors_kernel(float* d_u, float* d_errors, 
                                      int NLow, int MLow, int KLow,
                                      int N, int M, int K,
                                      float Lx, float Ly, float Lz,
                                      float t, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= N || j >= M || k >= K) return;

    int idx = i * M * K + j * K + k;

    float x = (i + NLow) * h;
    float y = (j + MLow) * h;
    float z = (k + KLow) * h;
    float analytical = analyticalSolution_device(Lx, Ly, Lz, x, y, z, t);

    float numerical = d_u[idx];

    d_errors[idx] = fabsf(numerical - analytical);
}

void compute_errors_device(float* d_u, float* d_errors,
                           int NLow, int MLow, int KLow,
                           int N, int M, int K,
                           float Lx, float Ly, float Lz,
                           float t, float h) {

    dim3 blockSize(8, 8, 4);
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (M + blockSize.y - 1) / blockSize.y,
        (K + blockSize.z - 1) / blockSize.z
    );
    compute_errors_kernel<<<gridSize, blockSize>>>(d_u, d_errors,
                                                  NLow, MLow, KLow,
                                                  N, M, K,
                                                  Lx, Ly, Lz, t, h);
    cudaDeviceSynchronize();
}

float find_max_host(float* h_array, int size) {
    float res = 0.0f, e;
    for (int i = 0; i < size; i++) {
        e = h_array[i];
        if (e > res)
            res = e;
    }
    return res;
}


void saveFloatsBinary(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file %s!\n", filename);
        return;
    }
    size_t written = fwrite(array, sizeof(float), size, file);
    fclose(file);
    if (written != size) {
        printf("Error: only wrote %zu/%d elements to %s\n", written, size, filename);
    }
}

void freeBuffersAndTerminate(float** hostBuffers, int numHostBuffers, float** deviceBuffers, int numDeviceBuffers, MPI_Comm cart_comm) {
    for (int i = 0; i < numHostBuffers; i++)
        if (hostBuffers[i] != NULL)
            cudaFreeHost(hostBuffers[i]);
    for (int i = 0; i < numDeviceBuffers; i++)
        if (deviceBuffers[i] != NULL)
            cudaFree(deviceBuffers[i]);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s NGlobal T L\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    float time_init;
    cudaEvent_t start, startLoop, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&startLoop);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int save = argc == 5 && argv[4][0] == 'a';
    if (rank == 0) {
        if (save) {
            struct stat st = {0};
            if (stat("floats", &st) == -1) {
                mkdir("floats", 0700);
            }
            if (stat("errors", &st) == -1) {
                mkdir("errors", 0700);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int NGlobal = atoi(argv[1]);
    float T = strtof(argv[2], NULL), L = strtof(argv[3], NULL);
    float h = L/(NGlobal-1), a2 = 1.0/9.0, tau = h / (sqrt(a2 * 12));
    int filenameLength = 100;
    char filename[filenameLength];

    int dims[DIMENTIONS] = {0, 0, 0};
    MPI_Dims_create(size, DIMENTIONS, dims);
    
    int periods[DIMENTIONS] = {1, 0, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, DIMENTIONS, dims, periods, 1, &cart_comm);
    
    int coords[DIMENTIONS];
    MPI_Cart_coords(cart_comm, rank, DIMENTIONS, coords);
    
    int left_rank, right_rank, front_rank, back_rank, down_rank, up_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &front_rank, &back_rank);
    MPI_Cart_shift(cart_comm, 2, 1, &down_rank, &up_rank);

    int N = NGlobal / dims[0];
    int M = NGlobal / dims[1];
    int K = NGlobal / dims[2];

    int NLow = coords[0] * N;
    int MLow = coords[1] * M;
    int KLow = coords[2] * K;
    
    if (coords[0] == dims[0]-1) N = NGlobal - NLow;
    if (coords[1] == dims[1]-1) M = NGlobal - MLow; 
    if (coords[2] == dims[2]-1) K = NGlobal - KLow;

    int uShape = N*M*K;

    float *h_ut = NULL, *h_errors = NULL;
    float *h_xPrev = NULL, *h_xPrevSend = NULL, *h_xNext = NULL, *h_xNextSend = NULL;
    float *h_yPrev = NULL, *h_yPrevSend = NULL, *h_yNext = NULL, *h_yNextSend = NULL;
    float *h_zPrev = NULL, *h_zPrevSend = NULL, *h_zNext = NULL, *h_zNextSend = NULL;

    cudaMallocHost(&h_ut, uShape * sizeof(float)); cudaMallocHost(&h_errors, uShape * sizeof(float));
    cudaMallocHost(&h_xPrev, M*K * sizeof(float)); cudaMallocHost(&h_xPrevSend, M*K * sizeof(float)); cudaMallocHost(&h_xNext, M*K * sizeof(float)); cudaMallocHost(&h_xNextSend, M*K * sizeof(float));
    
    cudaMallocHost(&h_yPrev, N*K * sizeof(float)); cudaMallocHost(&h_yPrevSend, N*K * sizeof(float)); cudaMallocHost(&h_yNext, N*K * sizeof(float)); cudaMallocHost(&h_yNextSend, N*K * sizeof(float));
    
    cudaMallocHost(&h_zPrev, N*M * sizeof(float)); cudaMallocHost(&h_zPrevSend, N*M * sizeof(float)); cudaMallocHost(&h_zNext, N*M * sizeof(float)); cudaMallocHost(&h_zNextSend, N*M * sizeof(float));

    float *d_errors = NULL, *d_ut0 = NULL, *d_ut1 = NULL, *d_xPrev = NULL, *d_xNext = NULL, *d_yPrev = NULL, *d_yNext = NULL, *d_zPrev = NULL, *d_zNext = NULL;
    cudaMalloc(&d_ut0, uShape * sizeof(float)); cudaMalloc(&d_ut1, uShape * sizeof(float)); cudaMalloc(&d_errors, uShape * sizeof(float));
    cudaMalloc(&d_xPrev, M*K * sizeof(float)); cudaMalloc(&d_xNext, M*K * sizeof(float)); cudaMalloc(&d_yPrev, N*K * sizeof(float)); cudaMalloc(&d_yNext, N*K * sizeof(float));
    cudaMalloc(&d_zPrev, N*M * sizeof(float)); cudaMalloc(&d_zNext, N*M * sizeof(float));

    float* hostBuffers[14] = {
        h_ut, h_errors, h_xPrev, h_xPrevSend, h_xNext, h_xNextSend, 
        h_yPrev, h_yPrevSend, h_yNext, h_yNextSend, 
        h_zPrev, h_zPrevSend, h_zNext, h_zNextSend
    };
    
    float* deviceBuffers[9] = {
        d_errors, d_ut0, d_ut1, d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext
    };

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % 2);

    int success = 1;
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    int globalSuccess;
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, cart_comm);
        return -1;
    }

    if (rank == 0) {
        printf("Grid: %dx%dx%d, Processes: %dx%dx%d\n", NGlobal, NGlobal, NGlobal, dims[0], dims[1], dims[2]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_init, start, stop);
    time_init /= 1000.0;

    float timeU0;
    cudaEventRecord(start, 0);

    makeU0_device(d_ut0, NLow, MLow, KLow, N, M, K, NGlobal, NGlobal, NGlobal, L, L, L, h);
    success = 1;
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, cart_comm);
        return -1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeU0, start, stop);
    timeU0 /= 1000.0;

    float timeExchanges;
    float timeExchange;
    cudaEventRecord(start, 0);
    
    exchange_cuda(d_ut0, N, M, K, cart_comm,
                  h_xPrev, h_xNext, h_yPrev, h_yNext, h_zPrev, h_zNext,
                  h_xPrevSend, h_xNextSend, h_yPrevSend, h_yNextSend, h_zPrevSend, h_zNextSend,
                  d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext,
                  left_rank, right_rank, front_rank, back_rank,
                  down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeExchanges, start, stop);
    timeExchanges /= 1000.0;

    float timeU1;
    cudaEventRecord(start, 0);
    makeU1_device(d_ut1, d_ut0, NLow, MLow, KLow, N, M, K, NGlobal, NGlobal, NGlobal, 
                          L, L, L, h, tau, a2,
                          d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext);

    success = 1;
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        success = 0;
    }
    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!globalSuccess) {
        if (rank == 0) {
            printf(cudaGetErrorString(cuda_err));
        }
        freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, cart_comm);
        return -1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeU1, start, stop);
    timeU1 /= 1000.0;

    float timeLoop;
    cudaEventRecord(startLoop, 0);
    int t = 2;
    
    float nextTime = tau * t;
    float* temp;
    float timeSteps = 0;
    float timeStep;
    for (; nextTime < T; ) {
        cudaEventRecord(start, 0);
        exchange_cuda(d_ut1, N, M, K, cart_comm,
                      h_xPrev, h_xNext, h_yPrev, h_yNext, h_zPrev, h_zNext,
                      h_xPrevSend, h_xNextSend, h_yPrevSend, h_yNextSend, h_zPrevSend, h_zNextSend,
                      d_xPrev, d_xNext, d_yPrev, d_yNext, d_zPrev, d_zNext,
                      left_rank, right_rank, front_rank, back_rank,
                      down_rank, up_rank, NLow, NGlobal, MLow, NGlobal, KLow, NGlobal);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeExchange, start, stop);
        timeExchanges += timeExchange / 1000.0;
        
        cudaEventRecord(start, 0);
        dim3 blockSize(4, 4, 4);
        dim3 gridSize(
            (N + blockSize.x - 1) / blockSize.x,
            (M + blockSize.y - 1) / blockSize.y,
            (K + blockSize.z - 1) / blockSize.z
        );
        
        step_kernel<<<gridSize, blockSize>>>(d_ut1, d_ut0, NLow, MLow, KLow,
                                            N, M, K, NGlobal, NGlobal, NGlobal,
                                            h, tau, a2,
                                            d_xPrev, d_xNext, d_yPrev, d_yNext,
                                            d_zPrev, d_zNext);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeStep, start, stop);
        timeSteps += timeStep / 1000.0;
        
        if (t % 10 == 0) {
            cudaMemcpy(h_ut, d_ut0, uShape * sizeof(float), cudaMemcpyDeviceToHost);
            compute_errors_device(d_ut0, d_errors, NLow, MLow, KLow, N, M, K, L, L, L, nextTime, h);
            cudaMemcpy(h_errors, d_errors, uShape * sizeof(float), cudaMemcpyDeviceToHost);
            float maxError = find_max_host(h_errors, N*M*K);
            float maxErrorGlobal;
            MPI_Reduce(&maxError, &maxErrorGlobal, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf("Analytical error at time %f: %.8f\n", nextTime, maxErrorGlobal);
            if (save) {
                snprintf(filename, filenameLength, "floats/floats_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, h_ut, uShape);
                snprintf(filename, filenameLength, "errors/errors_t%dNLow%dMLow%dKLow%d.bin", t, NLow, MLow, KLow);
                saveFloatsBinary(filename, h_errors, uShape);
            }
        }
        cuda_err = cudaGetLastError();
        success = (cuda_err == cudaSuccess);
        MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!globalSuccess) {
            if (rank == 0) {
                printf(cudaGetErrorString(cuda_err));
            }
            freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, cart_comm);
            return -1;
        }
        temp = d_ut1;
        d_ut1 = d_ut0;
        d_ut0 = temp;

        nextTime = tau * ++t;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeLoop, startLoop, stop);
    timeLoop /= 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(startLoop);
    cudaEventDestroy(stop);

    if (rank == 0) {
        printf("Simulation completed: %d time steps, Time: init-%f, U0-%f, exchanges-%f, U1-%f, steps-%f, Loop-%f, Full-%f\n", t, time_init, timeU0, timeExchanges, timeU1, timeSteps, timeLoop, time_init+timeU0+timeU1+timeLoop);
    }

    freeBuffersAndTerminate(hostBuffers, 14, deviceBuffers, 9, cart_comm);
    return 0;
}
