/*
  compare z = a*x + y for cuda and cpu
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<time.h>
#include<omp.h>

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a)<(b))?(a):(b))

void checksum(double *a, long n);
void list_env_properties();
__host__   void daxpy_omp(double *a, double *b, double *c, long n);
__global__ void daxpy(double *a, double *b, double *c, long n);
__global__ void daxpy2(double *a, double *b, double *c, long n);

int main(int argc, char **argv){
  double *a_h, *b_h, *c_h;  /* host version of vectors to sum and result */
  double *a,   *b,   *c;    /* device version of vectors to sum and result */

  long    n;                 /* problem size */
  long    i;                 /* counter */

  cudaEvent_t                /* CUDA timers */
    start_device,
    stop_device;  
  float time_device;

  double host_start, host_end;     /* host timers for OpenMP */

  int nblocks, nthreads_per_block, nt;

  n = atoi(argv[1]);                  /* size of arrays */
  nthreads_per_block = atoi(argv[2]); /* number of threads per block */
  nt = atoi(argv[3]);                 /* number of openmp threads */

  nblocks = MIN(n/nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

  /* creates CUDA timers but does not start yet */
  cudaEventCreate(&start_device);
  cudaEventCreate(&stop_device);

  /* allocate host memory */
  a_h = (double *) malloc(sizeof(double) * n);
  b_h = (double *) malloc(sizeof(double) * n);
  c_h = (double *) malloc(sizeof(double) * n);

  assert(a_h); assert(b_h); assert(c_h);

  /* allocate device memory */
  cudaMalloc((void **) &a, n*sizeof(double));
  cudaMalloc((void **) &b, n*sizeof(double));
  cudaMalloc((void **) &c, n*sizeof(double));

  /* initialize data on host */
  for (i=0;i<n;++i){
    b_h[i] =   1;
    c_h[i] =   0;
  }
  /* copy data to device memory */
  cudaMemcpy(b,b_h,n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c,c_h,n*sizeof(double), cudaMemcpyHostToDevice);

  printf("launching host kernel\n");
  omp_set_num_threads(nt);
  host_start = omp_get_wtime();    /* start openmp timers */
  daxpy_omp(a_h,b_h,c_h,n);
  host_end = omp_get_wtime();
  printf("time elapsed: %f\n", host_end-host_start);
  checksum(a_h,n);


  printf("launching kernel with %d blocks, %d threads per block\n", nblocks, nthreads_per_block);
  cudaEventRecord( start_device, 0 );  
  
  daxpy2<<<nblocks,nthreads_per_block>>>(a,b,c,n);
  
  cudaEventRecord( stop_device, 0 );
  cudaEventSynchronize( stop_device );
  cudaEventElapsedTime( &time_device, start_device, stop_device );

  cudaMemcpy(a_h,a,n*sizeof(double),cudaMemcpyDeviceToHost);

  checksum(a_h,n);

  printf("time elapsed device: %f(s)\n",  time_device/1000.);
  free(a_h); free(b_h), free(c_h);
  cudaFree(a);  cudaFree(b);  cudaFree(c);
  cudaEventDestroy( start_device );
  cudaEventDestroy( stop_device );
}


__host__ void daxpy_omp(double *a, double *b, double *c, long n){
  int i;
  double alpha = 1.2;
#pragma omp parallel for private(i) shared(n,a,b,c) 
  for (i=0;i<n;++i)
    a[i] = alpha*b[i] + c[i];
  return;
}
						   

__global__ void daxpy(double *a, double *b, double *c, long n){
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  double alpha = 1.2;
  if (tid < n)
    a[tid] = alpha*b[tid] + c[tid];
  return;
}

/* 
   "grid-stride" loop is best for memory coalescing 
*/
__global__ void daxpy2(double *a, double *b, double *c, long n){
  long i;
  double alpha = 1.2;

int tid0 = blockIdx.x * blockDim.x + threadIdx.x;


for (i = tid0; i < n; i += blockDim.x * gridDim.x) 
    a[i] = alpha*b[i] + c[i];

}


void checksum(double *a, long n){
  long i;
  double tot = 0;
  for (i=0;i<n;++i)
    tot += a[i];
  printf("checksum: %f\n", tot);
  return;
}

