#define Z 512
#define Y 512
#define X 512
#define TILE_IN_Z 4
#define TILE_IN_Y 8
#define TILE_IN_X 64
#define TILE_MID_Z 4
#define TILE_MID_Y 8
#define TILE_MID_X 64
#define TILE_SHARED_Z 4
#define TILE_SHARED_Y 8 
#define TILE_SHARED_X 32

#define OFF(x, y, z, nx, ny, nz) ((x) * (ny) * (nz) + (y) * (nz) + (z))

float gauss_seidel_gpu(double *u, double *b, double *res, int *num_iter, double norm0);

void test_residue(double * data,double *res);

void test_kernel(double * u,double *b);