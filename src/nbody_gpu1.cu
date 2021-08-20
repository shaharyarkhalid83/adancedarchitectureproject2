#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "newtimer.h"

#define BLOCK_SIZE 128
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
  initialize in memory as as:
  (particle 0:) x0 y0 z0 0.0 0.0 0.0
  (particle 1:) x1 y1 z1 0.0 0.0 0.0
  ....
*/
void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i+=6){
      for (int j=0;j<=2;++j){	
        data[i+j] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	data[i+j+3]=0;
      }
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;   /* p[i].x and p[j].x are generally far apart in memory */
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  FILE* datafile = NULL;  
  int nBodies;
  int nIters;
  int nt;
  nBodies = atoi(argv[1]);
  nIters  = atoi(argv[2]);
  nt      = atoi(argv[3]);
  const float dt = 0.01f; // time step

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  omp_set_num_threads(nt);
  srand(100);
  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Body *d_p = (Body*)d_buf;

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

    datafile = fopen("nbody.dat","w");  /* open output file */
//  fprintf(datafile,"%d %d %d\n", nBodies, nIters, 0);

  for (int iter = 1; iter <= nIters; iter++) {
    printf("iteration:%d\n", iter);

//    for (int i=0;i<nBodies;++i)
//      fprintf(datafile, "%f %f %f \n", p[i].x, p[i].y, p[i].z);

    StartTimer();

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);//copy data to GPU
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p, dt, nBodies); // compute interbody forces
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);//copy data back to CPU
    
    #pragma omp parallel for 
    for (int i = 0 ; i < nBodies; i++) { // integrate positions forward
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
  }
  fclose(datafile);
  double avgTime = totalTime / (double)(nIters-1); 

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);

  free(buf);
  cudaFree(d_buf);
}
