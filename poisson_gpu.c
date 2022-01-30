/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Date: 30/11/2021
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <emmintrin.h>
#include <immintrin.h>
#include "gauss_seidel.h"

#define N 512
#define TILE 256
#define K_Vec 1 
#define INNER_K 4
#define MAXITER 100
#define RTOL 1e-6
#define PADDING 2
#define SIDEPADDING 1
#define PI 3.14159265358979323846

void init_sol(double *__restrict__ b, double *__restrict__ u_exact, double *__restrict__ u)
{
    double a = N / 4.;
    double h = 1. / (N + 1);
#pragma omp parallel for
    for (int i = 0; i < N + PADDING; i++)
        for (int j = 0; j < N + PADDING; j++)
            for (int k = 0; k < N + PADDING; k++)
            {
                u[OFF(i,j,k,N+PADDING,N+PADDING,N+PADDING)] = 0.;
            }

#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                u_exact[OFF(i,j,k,N,N,N)] = sin(a * PI * i * h) * sin(a * PI * j * h) * sin(a * PI * k * h);
            }

#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                b[i * N * N + j * N + k] = 3. * a * a * PI * PI * sin(a * PI * (i + 1) * h) * sin(a * PI * (j + 1) * h) * sin(a * PI * (k + 1) * h) * h * h;
            }
}

double error(double *__restrict__ u, double *__restrict__ u_exact)
{
    double tmp = 0;
#pragma omp parallel for reduction(+ \
                                   : tmp)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp += pow((u_exact[OFF(i,j,k,N,N,N)] - u[OFF(i+SIDEPADDING,j+SIDEPADDING,k+SIDEPADDING, N+PADDING,N+PADDING,N+PADDING)]), 2);
            }
    double tmp2 = 0;
#pragma omp parallel for reduction(+ \
                                   : tmp2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                tmp2 += pow((u_exact[OFF(i,j,k,N,N,N)]), 2);
            }
    return pow(tmp, 0.5) / pow(tmp2, 0.5);
}

void gauss_seidel(double *__restrict__ u, double *__restrict__ b)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                u[(i + 1) * (N + PADDING) * (N + PADDING) + (j + 1) * (N + PADDING) + k + 1] =
                    (b[i * N * N + j * N + k]
                    + u[(i + 0) * (N + PADDING) * (N + PADDING) + (j + 1) * (N + PADDING) + k + 1]
                     + u[(i + 1) * (N + PADDING) * (N + PADDING) + (j + 0) * (N + PADDING) + k + 1]
                      + u[(i + 1) * (N + PADDING) * (N + PADDING) + (j + 1) * (N + PADDING) + k + 0]
                       + u[(i + 1) * (N + PADDING) * (N + PADDING) + (j + 1) * (N + PADDING) + k + 2]
                        + u[(i + 1) * (N + PADDING) * (N + PADDING) + (j + 2) * (N + PADDING) + k + 1]
                         + u[(i + 2) * (N + PADDING) * (N + PADDING) + (j + 1) * (N + PADDING) + k + 1]) / 6.0;
            }
        }
    }
}

double residual_norm(double *__restrict__ u, double *__restrict__ b)
{
    double norm = 0; 
#pragma omp parallel num_threads(8)
    {
        double norm2 = 0;
        int o = omp_get_thread_num();
        int io, jo, ko;
        io = ((o >> 2) & 1) * TILE;
        jo = ((o >> 1) & 1) * TILE;
        ko = (o & 1) * TILE;
        for (int ii = 0; ii < TILE; ii++)
        {
            for (int ji = 0; ji < TILE; ji++)
            {
                for (int ki = 0; ki < TILE; ki++)
                {
                    int i = io + ii, j = jo + ji, k = ko + ki;
                    double r = b[OFF(i, j, k, N, N, N)] + \
                            + u[OFF(i + SIDEPADDING + 1, j + SIDEPADDING, k + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] \
                            + u[OFF(i + SIDEPADDING - 1, j + SIDEPADDING, k + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] \
                            + u[OFF(i + SIDEPADDING, j + SIDEPADDING + 1, k + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] \
                            + u[OFF(i + SIDEPADDING, j + SIDEPADDING - 1, k + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] \
                            + u[OFF(i + SIDEPADDING, j + SIDEPADDING, k + SIDEPADDING + 1, N + PADDING, N + PADDING, N + PADDING)] \
                            + u[OFF(i + SIDEPADDING, j + SIDEPADDING, k + SIDEPADDING - 1, N + PADDING, N + PADDING, N + PADDING)] \
                            - 6.0 * u[OFF(i + SIDEPADDING, j + SIDEPADDING, k + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)];
                    norm2 += r * r;
                }
            }
        }
        #pragma omp atomic
        norm += norm2; 
    }
    return sqrt(norm);
}

int main(int argc, char **argv)
{
    double *u = (double *)malloc(sizeof(double) * (N + PADDING) * (N + PADDING) * (N + PADDING));
    double *u_exact = (double *)malloc(sizeof(double) * (N + PADDING) * (N + PADDING) * (N + PADDING));
    double *b = (double *)malloc(sizeof(double) * N * N * N);

    init_sol(b, u_exact, u);

    double normr0 = residual_norm(u, b);
    double normr = normr0;

    int tsteps = MAXITER;
    double time0 = omp_get_wtime();
    float elapsed_time = gauss_seidel_gpu(u, b, &normr, &tsteps, normr0);
    normr = residual_norm(u, b);
    if (normr < RTOL * normr0)
    {
        printf("Iteration %d, normr/normr0=%g\n", tsteps, normr / normr0);
        printf("Converged with %d iterations.\n", tsteps);
    }
    double time1 = omp_get_wtime() - time0;

    printf("time: %g\n", time1);
    printf("Residual norm: %g\n", normr);

    long residual_norm_bytes = sizeof(double) * ((N + PADDING) * (N + PADDING) * (N + PADDING) + (N * N * N)) * tsteps;
    long gs_bytes = sizeof(double) * ((N + PADDING) * (N + PADDING) * (N + PADDING) + 2 * (N * N * N)) * tsteps;

    long total_bytes = residual_norm_bytes + gs_bytes;
    double bandwidth = total_bytes / elapsed_time * 1000;

    printf("total bandwidth: %g GB/s\n", bandwidth / (double)(1 << 30));

    double relative_err = error(u, u_exact);
    printf("relative error: %g\n", relative_err);

    int num_threads = omp_get_max_threads();
    printf("openmp max num threads: %d\n", num_threads);

    free(u);
    free(u_exact);
    free(b);
    return 0;
}
