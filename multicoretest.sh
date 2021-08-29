# ! /bin/bash

./nbody_cpu_multicore 100000 20 1 > mtests.txt
./nbody_cpu_multicore 100000 20 2 >> mtests.txt
./nbody_cpu_multicore 100000 20 4 >> mtests.txt
./nbody_cpu_multicore 100000 20 8 >> mtests.txt
./nbody_cpu_multicore 100000 20 16 >> mtests.txt
./nbody_cpu_multicore 100000 20 24 >> mtests.txt
./nbody_cpu_multicore 100000 20 28 >> mtests.txt
./nbody_cpu_multicore 100000 20 32 >> mtests.txt
./nbody_cpu_multicore 100000 20 44 >> mtests.txt
./nbody_cpu_multicore 100000 20 56 >> mtests.txt
./nbody_cpu_multicore 100000 20 100 >> mtests.txt
./nbody_cpu_multicore 100000 20 150 >> mtests.txt
./nbody_cpu_multicore 100000 20 200 >> mtests.txt
./nbody_cpu_multicore 100000 20 300 >> mtests.txt
./nbody_cpu_multicore 100000 20 400 >> mtests.txt
./nbody_cpu_multicore 100000 20 500 >> mtests.txt
./nbody_cpu_multicore 100000 20 750 >> mtests.txt
./nbody_cpu_multicore 100000 20 1000 >> mtests.txt
./nbody_cpu_multicore 100000 20 2000 >> mtests.txt
./nbody_cpu_multicore 100000 20 3000 >> mtests.txt
./nbody_cpu_multicore 100000 20 4000 >> mtests.txt

