// Вариант 6
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>
#include <omp.h>

#define PI 3.14159265359

float analyticalSolution(float Lx, float Ly, float Lz, float x, float y, float z, float t) {
    float at = PI/3*sqrtf(4.0f/(Lx*Lx) + 1.0f/(Ly*Ly) + 4.0f/(Lz*Lz));
    return sinf(2.0f*PI*x/Lx) * sinf((1.0f + y/Ly) * PI) * sinf((1.0f + z/Lz) * 2.0f*PI) * cosf(at*t + PI);
}

float phi(float Lx, float Ly, float Lz, float x, float y, float z) {
    return analyticalSolution(Lx, Ly, Lz, x, y, z, 0.0f);
}

float aaLaplasian(float* ut, int N, int M, int K, int i, int j, int k, float h, float a2) {
    int mk = M*K;
    int jk = j*K;
    int imk = mk * i;

    int i_prev = (i == 0) ? N-2 : i-1;
    int i_next = (i == N-1) ? 1 : i+1;
    int j_prev = j-1;
    int j_next = j+1;
    int k_prev = (k == 0) ? K-2 : k-1;
    int k_next = (k == K-1) ? 1 : k+1;

    float d2x = ut[i_prev*mk + jk + k] - 2*ut[imk + jk + k] + ut[i_next*mk + jk + k];
    float d2y = ut[imk + j_prev*K + k] - 2*ut[imk + jk + k] + ut[imk + j_next*K + k];
    float d2z = ut[imk + jk + k_prev] - 2*ut[i*mk + jk + k] + ut[imk + jk + k_next];

    return a2 * (d2x + d2y + d2z) / (h * h);
}

float* makeU0(int N, int M, int K, float Lx, float Ly, float Lz, float h) {
    float* res = calloc(N*M*K, sizeof(float));
    if (res == NULL) {
        printf("Error creating buffer in U1, terminating...\n");
        return NULL;
    }
    float x, y, z;
    int idxi, idxj;

    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        x = i * h;
        for (int j = 0; j < M; j++) {
            y = j * h;
            idxj = idxi + j*K;
            for (int k = 0; k < K; k++) {
                z = k * h;
                res[idxj+k] = ((j == 0) || (j == M-1)) ? 0.0 : phi(Lx, Ly, Lz, x, y, z);
            }
        }
    }
    return res;
}

float* makeU1(float* ut0, int N, int M, int K, float Lx, float Ly, float Lz, float h, float tau, float a2) {
    float* res = calloc(N*M*K, sizeof(float));
    if (res == NULL) {
        printf("Error creating buffer in U1, terminating...\n");
        return NULL;
    }
    int idxi, idxj, jk;
    float l;

    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        for (int j = 1; j < M-1; j++) {
            jk = j*K;
            idxj = idxi + jk;
            for (int k = 0; k < K; k++) {
                l = tau * tau * aaLaplasian(ut0, N, M, K, i, j, k, h, a2) / 2;
                res[idxj+k] = ut0[idxj + k] + l;
            }
        }
    }
    return res;
}

void step(float** ut1_ptr, float** ut0_ptr, int N, int M, int K, float h, float tau, float a2) {
    float* ut1 = *ut1_ptr;
    float* ut0 = *ut0_ptr;
    int idxi, idxj, jk;
    float l, dt2;

    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        for (int j = 1; j < M-1; j++) {
            jk = j*K;
            idxj = idxi + jk;
            for (int k = 0; k < K; k++) {
                l = tau * tau * aaLaplasian(ut1, N, M, K, i, j, k, h, a2);
                dt2 = 2*ut1[idxj+k] - ut0[idxj+k];
                ut0[idxj+k] = dt2 + l;
            }
        }
    }
    float* swap = *ut1_ptr;
    *ut1_ptr = *ut0_ptr;
    *ut0_ptr = swap;
}

void saveFloatsBinary(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    fwrite(array, sizeof(float), size, file);
    fclose(file);
}

void checkAgainstAnalytical(float* u, float* errors, int N, int M, int K, float Lx, float Ly, float Lz, float t, float h) {
    float maxError = 0.0f, x, y, z, analytical, numerical, oneError;
    int idx;

    for (int i = 0; i < N; i++) {
        x = i * h;
        for (int j = 0; j < M; j++) {
            y = j * h;
            for (int k = 0; k < K; k++) {
                z = k * h;
                idx = i*M*K + j*K + k;
                analytical = analyticalSolution(Lx, Ly, Lz, x, y, z, t);
                numerical = u[idx];
                oneError = fabs(numerical - analytical);
                errors[idx] = oneError;
                if (oneError > maxError) maxError = oneError;
            }
        }
    }
    printf("Analytical error at time %f: %.8f\n", t, maxError);
}


int main(int argc, char *argv[]) {
    if ((argc < 4) || (argc > 5)) {
        printf("Provide parameters for N, T, L.\n");
        return -1;
    }

    int N = atoi(argv[1]), uShape = N*N*N, t = 0;
    printf("Number of nodes in one direction: %d\n", N);
    float L = strtof(argv[3], NULL), T = strtof(argv[2], NULL);
    float h = L/(N-1), a2 = 1.0/9.0, tau = h / (sqrt(a2 * 12));
    int filenameLength = 100;
    char filename[100];

    int save = argc == 5 && argv[4][0] == 'a';
    if (save) {
        struct stat st = {0};
        if (stat("floats", &st) == -1) {
            mkdir("floats", 0700);
        }
        if (stat("errors", &st) == -1) {
            mkdir("errors", 0700);
        }
    }
    double start_time = omp_get_wtime();

    float* ut0 = makeU0(N, N, N, L, L, L, h);
    float* errors = calloc(uShape, sizeof(float));
    if (ut0 == NULL || errors == NULL) {
        return -1;
    }
    if (save) {
        saveFloatsBinary("floats/floats0.bin", ut0, uShape);
        saveFloatsBinary("errors/errors0.bin", errors, uShape);
    }
    t++;

    float* ut1 = makeU1(ut0, N, N, N, L, L, L, h, tau, a2);
    if (ut1 == NULL) {
        return -1;
    }
    t++;

    for (; tau * t < T; t++) {
        step(&ut1, &ut0, N, N, N, h, tau, a2);
        
        if (t % 10 == 0) {
            checkAgainstAnalytical(ut1, errors, N, N, N, L, L, L, tau*t, h);

            if (save) {
                snprintf(filename, filenameLength, "floats/floats%d.bin", t);
                saveFloatsBinary(filename, ut1, uShape);
                snprintf(filename, filenameLength, "errors/errors%d.bin", t);
                saveFloatsBinary(filename, errors, uShape);
            }
        }
    }

    double time = omp_get_wtime() - start_time;
    printf("Simulation completed in %f seconds\n", time);

    free(ut0);
    free(ut1);

    FILE* file = fopen("times", "a");
    if (file == NULL) {
        printf("Error opening file times!\n");
    } else {
        char record[100];
        int chars_written = snprintf(record, 100, "no_parallel,%.3f,%d,%.2f,,\n", time, N, L);
        if (chars_written > 0 && chars_written < 100) {
            fwrite(record, sizeof(char), chars_written, file);
        }
        fclose(file);
    }
    return 0;
}
