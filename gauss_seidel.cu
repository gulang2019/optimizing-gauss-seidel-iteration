#include <math.h>
#include <stdio.h>
#define N 512
#define Z 512
#define Y 512
#define X 512
#define TILE_IN_Z 8
#define TILE_IN_Y 8
#define TILE_IN_X 32
#define TILE_SHARE_Z 8
#define TILE_SHARE_Y 8
#define TILE_SHARE_X 32
#define TILE_REG_Z 8
#define TILE_REG_Y 16
#define TILE_REG_X 32
#define TILE_MID_Z 128
#define TILE_MID_Y 128
#define TILE_MID_X 128
#define PADDING 2
#define SIDEPADDING 1
#define a (N / 4)
#define h ()
#define OFF(x, y, z, nx, ny, nz) ((x) * (ny) * (nz) + (y) * (nz) + (z))
#define RTOL 1e-6
#define blockSize 256
#define MAXITER 100
#define WARPSIZE 32
extern "C" float gauss_seidel_gpu(double *u, double *b, double *res, int *num_iter, double norm0);

extern "C" void test_residue(double *data, double *res);

extern "C" void test_kernel(double *u, double *b);

// sum stores the residue cube
__global__ void gauss_seidel_kernel_(double *u, double *b, double *sum)
{
    int x_i = threadIdx.x,
        y_i = threadIdx.y,
        z_i = threadIdx.z;
    int x_base = TILE_MID_X * blockIdx.x, 
        y_base = TILE_MID_Y * blockIdx.y,
        z_base = TILE_MID_Z * blockIdx.z;
    // black or red offset on x
    int x_i_b = ((y_i ^ z_i) & 1) + (x_i << 1);
    int x_i_r = !((y_i ^ z_i) & 1) + (x_i << 1);
    __shared__ double us[TILE_SHARE_Z][TILE_SHARE_Y][TILE_SHARE_X / 2];
    const int d[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    double reg[2][TILE_REG_Z / TILE_IN_Z][TILE_REG_Y / TILE_IN_Y][TILE_REG_X / TILE_IN_X];
    for(int x_reg = 0; x_reg < TILE_MID_X; x_reg += TILE_REG_X)
        for(int y_reg = 0; y_reg < TILE_MID_Y; y_reg += TILE_REG_Y)
            for(int z_reg = 0; z_reg < TILE_MID_Z; z_reg += TILE_REG_Z)
            {
                /*1. Init: global to reg*/
                for(int x_m = 0; x_m < TILE_REG_X; x_m += TILE_IN_X)
                    for(int y_m = 0; y_m < TILE_REG_Y; y_m += TILE_IN_Y)
                        for(int z_m = 0; z_m < TILE_REG_Z; z_m += TILE_IN_Z)
                        {
                            int x_o = x_base + x_reg,
                                y_o = y_base + y_reg,
                                z_o = z_base + z_reg;
                            
                            reg[1][z_m / TILE_IN_Z][y_m / TILE_IN_Y][x_m / TILE_IN_X] =\
                                            u[OFF(z_o + z_m + z_i + SIDEPADDING, y_o + y_m + y_i + SIDEPADDING, x_o + x_m + x_i_b + SIDEPADDING,
                                                Z + PADDING, Y + PADDING, X + PADDING)];
                            reg[0][z_m / TILE_IN_Z][y_m / TILE_IN_Y][x_m / TILE_IN_X] =\
                                            u[OFF(z_o + z_m + z_i + SIDEPADDING, y_o + y_m + y_i + SIDEPADDING, x_o + x_m + x_i_r + SIDEPADDING,
                                Z + PADDING, Y + PADDING, X + PADDING)];
                        
                        }
                /*RED&BLACK: calculation*/
                for(int x_share = 0; x_share < TILE_REG_X; x_share += TILE_SHARE_X)
                    for(int y_share = 0; y_share <TILE_REG_Y; y_share += TILE_SHARE_Y)
                        for(int z_share = 0; z_share < TILE_REG_Z; z_share += TILE_SHARE_Z)
                        {
                            int x_o = x_base + x_reg + x_share,
                                y_o = y_base + y_reg + y_share,
                                z_o = z_base + z_reg + z_share;
                            /*2. BLACK: reg to shared*/
                            for (int x_m = 0; x_m < TILE_SHARE_X; x_m += TILE_IN_X)
                                for (int y_m = 0; y_m < TILE_SHARE_Y; y_m += TILE_IN_Y)
                                    for (int z_m = 0; z_m < TILE_SHARE_Z; z_m += TILE_IN_Z)
                                        us[z_m + z_i][y_m + y_i][(x_m >> 1) + x_i] =\
                                        reg[1][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X];
                            __syncthreads();                            
                            /*3. RED: calculation*/
                            for (int x_m = 0; x_m < TILE_SHARE_X; x_m += TILE_IN_X)
                                for (int y_m = 0; y_m < TILE_SHARE_Y; y_m += TILE_IN_Y)
                                    for (int z_m = 0; z_m < TILE_SHARE_Z; z_m += TILE_IN_Z)
                                    {
                                        double acc = b[OFF(z_o + z_m + z_i, y_o + y_m + y_i, x_o + x_m + x_i_r, Z, Y, X)];
                                        for (int dir = 0; dir < 6; dir++)
                                        {
                                            int x_in = x_m + x_i_r + d[dir][0],
                                                y_in = y_m + y_i + d[dir][1],
                                                z_in = z_m + z_i + d[dir][2];
                                            double tmp;
                                            if (x_in < 0 || y_in < 0 || z_in < 0 || x_in >= TILE_SHARE_X || y_in >= TILE_SHARE_Y || z_in >= TILE_SHARE_Z)
                                            {
                                                int x = x_in + x_o,
                                                    y = y_in + y_o,
                                                    z = z_in + z_o;
                                                tmp = u[OFF(z + SIDEPADDING, y + SIDEPADDING, x + SIDEPADDING,
                                                            Z + PADDING, Y + PADDING, X + PADDING)];
                                            }
                                            else
                                                tmp = us[z_in][y_in][x_in >> 1];
                                            acc += tmp;
                                        }
                                        /*RED: residue calculation*/
                                        double res = 6.0 * reg[0][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X] -
                                                    acc;
                                        res *= res;
                                        sum[OFF(z_o + z_m + z_i, y_o + y_m + y_i, x_o + x_m + x_i_r,
                                                Z, Y, X)] = res;
                                        reg[0][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X] = acc / 6.0;
                                    }
                            __syncthreads();
                            /*4. RED: reg to shared*/
                            for (int x_m = 0; x_m < TILE_SHARE_X; x_m += TILE_IN_X)
                                for (int y_m = 0; y_m < TILE_SHARE_Y; y_m += TILE_IN_Y)
                                    for (int z_m = 0; z_m < TILE_SHARE_Z; z_m += TILE_IN_Z)
                                    {
                                        us[z_m + z_i][y_m + y_i][(x_m >> 1) + x_i] =
                                            reg[0][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X];
                                    }
                            __syncthreads();
                            /*5. BLACK: calculation*/
                            for (int x_m = 0; x_m < TILE_SHARE_X; x_m += TILE_IN_X)
                                for (int y_m = 0; y_m < TILE_SHARE_Y; y_m += TILE_IN_Y)
                                    for (int z_m = 0; z_m < TILE_SHARE_Z; z_m += TILE_IN_Z)
                                    {
                                        double acc = b[OFF(z_o + z_m + z_i, y_o + y_m + y_i, x_o + x_m + x_i_b, Z, Y, X)];
                                        for (int dir = 0; dir < 6; dir++)
                                        {
                                            int x_in = x_m + x_i_b + d[dir][0],
                                                y_in = y_m + y_i + d[dir][1],
                                                z_in = z_m + z_i + d[dir][2];
                                            double tmp;
                                            if (x_in < 0 || y_in < 0 || z_in < 0 || x_in >= TILE_SHARE_X || y_in >= TILE_SHARE_Y || z_in >= TILE_SHARE_Z)
                                            {
                                                int x = x_in + x_o,
                                                    y = y_in + y_o,
                                                    z = z_in + z_o;

                                                tmp = u[OFF(z + SIDEPADDING, y + SIDEPADDING, x + SIDEPADDING,
                                                            Z + PADDING, Y + PADDING, X + PADDING)];
                                            }
                                            else
                                                tmp = us[z_in][y_in][x_in >> 1];
                                            acc += tmp;
                                        }
                                        /*5. BLACK: calculation*/
                                        double res = 6.0 * reg[1][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X] -
                                                    acc;
                                        res *= res;
                                        sum[OFF(z_o + z_m + z_i, y_o + y_m + y_i, x_o + x_m + x_i_b,
                                                Z, Y, X)] = res;
                                        reg[1][(z_m + z_share) / TILE_IN_Z][(y_m + y_share) / TILE_IN_Y][(x_m + x_share) / TILE_IN_X] = acc / 6.0;
                                    }
                            __syncthreads();
                        }
                /*6. RED&BLACK: store back to global memory*/
                for(int x_m = 0; x_m < TILE_REG_X; x_m += TILE_IN_X)
                    for(int y_m = 0; y_m < TILE_REG_Y; y_m += TILE_IN_Y)
                        for(int z_m = 0; z_m < TILE_REG_Z; z_m += TILE_IN_Z)
                        {
                            int x_o = x_base + x_reg,
                                y_o = y_base + y_reg,
                                z_o = z_base + z_reg;
                            
                            u[OFF(z_o + z_m + z_i + SIDEPADDING, y_o + y_m + y_i + SIDEPADDING, x_o + x_m + x_i_b + SIDEPADDING,
                                                Z + PADDING, Y + PADDING, X + PADDING)] =\
                                            reg[1][z_m / TILE_IN_Z][y_m / TILE_IN_Y][x_m / TILE_IN_X];
                            u[OFF(z_o + z_m + z_i + SIDEPADDING, y_o + y_m + y_i + SIDEPADDING, x_o + x_m + x_i_r + SIDEPADDING,
                                Z + PADDING, Y + PADDING, X + PADDING)] =\
                                            reg[0][z_m / TILE_IN_Z][y_m / TILE_IN_Y][x_m / TILE_IN_X];
                        }      
            }
}


__global__ void residue1(double *u, double *b, double *sum)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    double tmp = b[OFF(z, y, x, N, N, N)] +
                 u[OFF(z + SIDEPADDING + 1, y + SIDEPADDING, x + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] +
                 u[OFF(z + SIDEPADDING - 1, y + SIDEPADDING, x + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] +
                 u[OFF(z + SIDEPADDING, y + SIDEPADDING + 1, x + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] +
                 u[OFF(z + SIDEPADDING, y + SIDEPADDING - 1, x + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)] +
                 u[OFF(z + SIDEPADDING, y + SIDEPADDING, x + SIDEPADDING + 1, N + PADDING, N + PADDING, N + PADDING)] +
                 u[OFF(z + SIDEPADDING, y + SIDEPADDING, x + SIDEPADDING - 1, N + PADDING, N + PADDING, N + PADDING)] -
                 6.0 * u[OFF(z + SIDEPADDING, y + SIDEPADDING, x + SIDEPADDING, N + PADDING, N + PADDING, N + PADDING)];
    sum[OFF(z, y, x, N, N, N)] = tmp * tmp;
}

__device__ void warpReduce(volatile double *sdata, unsigned int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void residue(double *sum, double *res)
{
    __shared__ double sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    sdata[tid] = sum[i] + sum[i + blockSize];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        res[blockIdx.x] = sdata[0];
}
float gauss_seidel_gpu(double *u, double *b, double *res, int *num_iter, double norm0)
{
    double *d_u;
    double *d_b;
    double *d_s;
    double *d_res1, *d_res2, *d_res3;
    size_t sz_u = (N + PADDING) * (N + PADDING) * (N + PADDING) * sizeof(double);
    size_t sz_b = (N) * (N) * (N) * sizeof(double);
    size_t sz_s = N * N * N * sizeof(double);
    cudaMalloc(((void **)&d_u), sz_u);
    cudaMalloc(((void **)&d_b), sz_b);
    cudaMalloc((void **)&d_s, sz_s);
    cudaMalloc((void **)&d_res1, N * N * sizeof(double));
    cudaMalloc((void **)&d_res2, N * sizeof(double));
    cudaMalloc((void **)&d_res3, sizeof(double));
    cudaMemcpy(d_u, u, sz_u, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sz_b, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float elapsed_time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 grid_dim(N / TILE_MID_X, N / TILE_MID_Y, N / TILE_MID_Z);
    dim3 block_dim(TILE_IN_X / 2, TILE_IN_Y, TILE_IN_Z);
    printf("Mid Tiling: %d %d %d.\n", TILE_MID_X, TILE_MID_Y, TILE_MID_Z);
    printf("Inner Tiling(block): %d %d %d\n", TILE_IN_X, TILE_IN_Y, TILE_IN_Z);
    printf("grid  dim:  %d, %d, %d.\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d.\n", block_dim.x, block_dim.y, block_dim.z);
    bool converged = 0;
    int iter;

    for (iter = 0; iter < MAXITER; iter++)
    {
        gauss_seidel_kernel_<<<grid_dim, block_dim>>>(d_u, d_b, d_s);
        residue<<<N * N, blockSize>>>(d_s, d_res1);
        residue<<<N, blockSize>>>(d_res1, d_res2);
        residue<<<1, blockSize>>>(d_res2, d_res3);
        cudaMemcpy(res, d_res3, sizeof(double), cudaMemcpyDeviceToHost);
        *res = sqrt(*res);
        printf("iter %d, res %f\n", iter, *res);
        if (!converged && *res < RTOL * norm0)
        {
            *num_iter = iter + 1;
            converged = 1;
        }
        if (converged && iter >= 34)
            break;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("norm0 %f, normr %f\n", norm0, *res);
    printf("elapsed time %.5f, normr/norm0 %f, time/iter %f\n", elapsed_time / 1000, *res / norm0, elapsed_time / iter);

    cudaMemcpy(u, d_u, sz_u, cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    cudaFree(d_b);
    cudaFree(d_s);
    cudaFree(d_res1);
    cudaFree(d_res2);
    cudaFree(d_res3);
    return elapsed_time;
}