#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<omp.h>
#include "timer.h"

#ifdef PAPI
  #include<string.h>
  #include<papi.h>
#endif

void handle_error(int errcode);

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  
#pragma omp parallel for schedule(dynamic)

  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  #ifdef PAPI
  double cycles_per_sec;
  int num_events, n, ret;
  char *event_name;
  //int Events[] = {PAPI_L1_DCM};
  int Events[] = {PAPI_L2_DCA, PAPI_L2_LDM};

  long_long *values;               
  long_long ts, tf;                
  num_events = sizeof(Events)/sizeof(int);
  event_name = (char *) malloc(128);
  values = (long_long *) malloc(num_events*sizeof(long_long));

  #endif

  FILE *datafile;  
  int nBodies = 100000;
  int nthreads = 28;
  int nIters = 20; 

  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nIters = atoi(argv[2]);
  if (argc > 3) nthreads = atoi(argv[3]);

  const float dt = 0.01f; 

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  omp_set_num_threads(nthreads);
  randomizeBodies(buf, 6*nBodies); 

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    printf("iteration:%d\n", iter);
    
    StartTimer();

    bodyForce(p, dt, nBodies);         

    for (int i = 0 ; i < nBodies; i++) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) {                      
      totalTime += tElapsed; 
    }
  }
  
  double avgTime = totalTime / (double)(nIters-1); 

  printf("avgTime: %f   totTime: %f \n", avgTime, totalTime);


  #ifdef PAPI
  
  if ((ret=PAPI_start_counters(Events, num_events)) != PAPI_OK)
  handle_error(ret);
  ts = PAPI_get_real_usec();
 
  bodyForce(p, dt, nBodies);        
  
  for (int i = 0 ; i < nBodies; i++) { 
    p[i].x += p[i].vx*dt;
    p[i].y += p[i].vy*dt;
    p[i].z += p[i].vz*dt;
  }

  tf = PAPI_get_real_usec();       

  if ((ret=PAPI_stop_counters(values, num_events)) != PAPI_OK)
    handle_error(ret);


  for (int i=0;i<num_events;++i){       /* print name/value of each event counter */
    PAPI_event_code_to_name(Events[i],event_name);
    printf("%s:%lld\n", event_name, values[i]);
    if (strcmp(event_name,"PAPI_TOT_CYC") == 0){
      cycles_per_sec = (double) values[i]/(double) ((tf-ts)*1.e-6);
      printf("cycles per second: %e\n", cycles_per_sec);
    }
  }
  printf("tot time: %e n", (tf-ts)/1.e6);

#endif

  free(buf);
}


void handle_error(int errcode){
  char error_str[PAPI_MAX_STR_LEN];

  fprintf(stderr,"PAPI_error in call to create_eventset %d: %s\n",errcode,error_str);
  exit(1);
} 
