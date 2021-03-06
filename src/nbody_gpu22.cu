#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "newtimer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float4 *pos, *vel; } BodySystem;

/*
separate positions from velocities
*/

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  StartTimer();
  int nBodies = 100000;
  int nIters = 20;
  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nIters = atoi(argv[2]); 
 
  const float dt = 0.01f; // time step
  
  int bytes = 2*nBodies*sizeof(float4);
  float *buf = (float*)malloc(bytes);
  BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  double tStartLoop = 0.0;
  double tEndLoop = 0.0;
  double loopTime  = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    printf("iteration:%d\n", iter);  
    
    tStartLoop = GetTimer() / 1000.0;
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies);
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    tEndLoop = GetTimer() / 1000.0;
    loopTime += tEndLoop - tStartLoop; 

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }



  }


  free(buf);
  cudaFree(d_buf);
  const double tEndTime = GetTimer() / 1000.0;
  printf("percent of time in bodyForce: %f   \n", loopTime/tEndTime);
}

