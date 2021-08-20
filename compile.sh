#!/bin/bash
set echo on
gcc -fopenmp -DOMP -O2 -std=c11 -c  ./src/nbody_cpu_serial.c 
gcc -O2 -std=c11 -c  ./src/timer.c 
gcc -fopenmp -O2 -o nbody_cpu_serial nbody_cpu_serial.o timer.o -lm 
nvcc -Xcompiler -fopenmp ./src/nbody_gpu1.cu -o nbody_gpu1 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu2.cu -o nbody_gpu2 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu3.cu -o nbody_gpu3 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu4.cu -o nbody_gpu4 -lm
