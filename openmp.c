// Вариант 6
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> 
#include <math.h>
#include <errno.h>
#include <omp.h>

#define PI 3.14159265359

float analyticalSolution(float Lx, float Ly, float Lz, float x, float y, float z, float t) {
    float at = PI/3*sqrt(4/(Lx*Lx) + 1/(Ly*Ly) + 4/(Lz*Lz));
    return sin(2*PI*x/Lx) * sin((1 + y/Ly) * PI) * sin((1 + z/Lz) * 2*PI) * cos(at*t + PI);
}

float phi(float Lx, float Ly, float Lz, float x, float y, float z) {
    return analyticalSolution(Lx, Ly, Lz, x, y, z, 0.0);
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

void swapBuffers(float* b0, float* b1, int N, int M, int K) {
    float tmp;
    int idxi, idxj;
    #pragma omp parallel for private(idxi, idxj, tmp)
    for (int i = 0; i < N; i++) {
        idxi = i*M*K;
        for (int j = 0; j < M; j++) {
            idxj = idxi + j*K;
            for (int k = 0; k < K; k++) {
                tmp = b0[idxj + k];
                b0[idxj + k] = b1[idxj + k];
                b1[idxj + k] = tmp;
            }
        }
    }
}

float* makeU0(int N, int M, int K, float Lx, float Ly, float Lz, float h) {
    float* res = calloc(N*M*K, sizeof(float));
    float x, y, z;
    int idxi, idxj;
    
    #pragma omp parallel for private(idxi, idxj, x, y, z)
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
    float* res = makeU0(N, M, K, Lx, Ly, Lz, h);
    int idxi, idxj, jk;
    float l;
    
    #pragma omp parallel for private(idxi, idxj, jk, l)
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

void step(float* ut1, float* ut0, int N, int M, int K, float h, float tau, float a2) {
    int idxi, idxj, jk;
    float l, dt2;
    
    #pragma omp parallel for private(idxi, idxj, jk, l, dt2)
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
    swapBuffers(ut0, ut1, N, M, K);
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

void checkAgainstAnalytical(float* u, int N, int M, int K, float Lx, float Ly, float Lz, float t, float h) {
    float max_error = 0.0f, x, y, z, analytical, numerical, error;

    #pragma omp parallel for reduction(max:max_error) private(x, y, z, analytical, numerical, error)
    for (int i = 0; i < N; i++) {
        x = i * h;
        for (int j = 1; j < M-1; j++) {
            y = j * h;
            for (int k = 0; k < K; k++) {
                z = k * h;
                analytical = analyticalSolution(Lx, Ly, Lz, x, y, z, t);
                numerical = u[i*M*K + j*K + k];
                error = fabs(numerical - analytical);
                if (error > max_error) max_error = error;
            }
        }
    }
    printf("Analytical max error at t %f: %f\n", t, max_error);
}

void checkBoundaryConsistency(float* u, int N, int M, int K, int step) {
    float max_diff_x = 0.0f, max_diff_z = 0.0f, diff;
    int lastI = (N-1)*M*K, jK, iMK, MK = M*K;

    #pragma omp parallel for reduction(max:max_diff_x) private(lastI, jK)
    for (int j = 0; j < M; j++) {
        jK = j*K;
        for (int k = 0; k < K; k++) {
            diff = fabs(u[lastI + jK + k] - u[jK + k]);
            if (diff > max_diff_x) max_diff_x = diff;
        }
    }
 
    #pragma omp parallel for reduction(max:max_diff_z) private(lastI, iMK, MK, jK)
    for (int i = 0; i < N; i++) {
        iMK = i*MK;
        for (int j = 0; j < M; j++) {
            jK = j*K;
            diff = fabs(u[iMK + jK + (K-1)] - u[i*M*K + jK]);
            if (diff > max_diff_z) max_diff_z = diff;
        }
    }
    printf("Step %d: Max boundary differences - x: %e, z: %e\n", step, max_diff_x, max_diff_z);
}

int main(int argc, char *argv[]) {
    if ((argc < 4) || (argc > 5)) {
        printf("Provide parameters for N, T, L.\n");
        return -1;
    }

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    printf("Using OpenMP with %d threads\n", max_threads);

    int N = atoi(argv[1]), uShape = N*N*N, t = 0;
    printf("Number of nodes in one direction: %d", N);
    float L = strtof(argv[3], NULL), T = strtof(argv[2], NULL);
    float h = L/(N-1), a2 = 1.0/9.0, tau = h / (sqrt(a2 * 3) * 4.0);
    int filenameLength = 100;
    char filename[filenameLength];

    if (argc == 5) {
        int status = mkdir("floats", 0755);
        if (status != 0) {
            printf("Remove floats folder before usage.");
            return -1;
        }
    }

    double start_time = omp_get_wtime();

    float* ut0 = makeU0(N, N, N, L, L, L, h);
    if (argc == 5)
        saveFloatsBinary("floats/floats00.bin", ut0, uShape);
    t++;

    float* ut1 = makeU1(ut0, N, N, N, L, L, L, h, tau, a2);
    t++;

    for (; tau * t < T; t++) {
        step(ut1, ut0, N, N, N, h, tau, a2);
        if (t % 10 == 0) {
            checkAgainstAnalytical(ut1, N, N, N, L, L, L, tau*t, h);
            checkBoundaryConsistency(ut1, N, N, N, t);

            if (argc == 5) {
                snprintf(filename, filenameLength, "floats/floats%02d.bin", t);
                saveFloatsBinary(filename, ut1, uShape);
            }
        }
    }

    double time = omp_get_wtime() - start_time;
    printf("Simulation completed in %.2f seconds\n", time);

    free(ut0);
    free(ut1);
    return 0;
}
