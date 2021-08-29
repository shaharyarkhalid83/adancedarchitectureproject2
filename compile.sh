#!/bin/bash
set echo on
gcc -fopenmp -DOMP -DPAPI -O2 -std=c11 -c  ./src/nbody_cpu_serial.c
gcc -O2 -std=c11 -c  ./src/timer.c
gcc -fopenmp -O2 -o nbody_cpu_serial nbody_cpu_serial.o timer.o -lm -lpapi
gcc -fopenmp -DPAPI -O2 -std=c11 -c  ./src/nbody_cpu_multicore.c
gcc -fopenmp -O2 -o nbody_cpu_multicore nbody_cpu_multicore.o timer.o -lm -lpapi
nvcc -Xcompiler -fopenmp ./src/nbody_gpu1.cu -o nbody_gpu1 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu2.cu -o nbody_gpu2 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu3.cu -o nbody_gpu3 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu4.cu -o nbody_gpu4 -lm


gcc -fopenmp -DOMP -O2 -std=c11 -c  ./src/nbody_cpu_serial2.c
gcc -fopenmp -O2 -o nbody_cpu_serial2 nbody_cpu_serial2.o timer.o -lm -lpapi
gcc -fopenmp -DPAPI -O2 -std=c11 -c  ./src/nbody_cpu_multicore2.c
gcc -fopenmp -O2 -o nbody_cpu_multicore2 nbody_cpu_multicore2.o timer.o -lm -lpapi
nvcc -Xcompiler -fopenmp ./src/nbody_gpu12.cu -o nbody_gpu12 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu22.cu -o nbody_gpu22 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu32.cu -o nbody_gpu32 -lm
nvcc -Xcompiler -fopenmp ./src/nbody_gpu42.cu -o nbody_gpu42 -lm

icc -fopenmp -DOMP -DPAPI -qopt-report -O2 -std=c11 -o nbody_cpu_icc_serial.o -c  ./src/nbody_cpu_serial.c
#icc -O2 -o timer_icc.o -c  ./src/timer.c
icc -fopenmp -O2 -o nbody_cpu_icc_serial nbody_cpu_icc_serial.o timer.o -lm -lpapi
icc -fopenmp -DPAPI -O2 -std=c11 -o nbody_cpu_icc_multicore.o -c  ./src/nbody_cpu_multicore.c
icc -fopenmp -O2 -o nbody_cpu_icc_multicore nbody_cpu_icc_multicore.o timer.o -lm -lpapi

icc -fopenmp -DOMP -O2 -std=c11  -o nbody_cpu_icc_serial2.o -c  ./src/nbody_cpu_serial2.c
icc -fopenmp -O2 -o nbody_cpu_icc_serial2 nbody_cpu_icc_serial2.o timer.o -lm -lpapi
icc -fopenmp -DPAPI -O2 -std=c11 -o nbody_cpu_icc_multicore2.o -c  ./src/nbody_cpu_multicore2.c
icc -fopenmp -O2 -o nbody_cpu_icc_multicore2 nbody_cpu_icc_multicore2.o timer.o -lm -lpapi

