# ! /bin/bash

./nbody_cpu_multicore 1000 20 1 > wtests.txt
./nbody_cpu_multicore 2000 20 2 >> wtests.txt
./nbody_cpu_multicore 4000 20 4 >> wtests.txt
./nbody_cpu_multicore 8000 20 8 >> wtests.txt
./nbody_cpu_multicore 16000 20 16 >> wtests.txt
./nbody_cpu_multicore 24000 20 24 >> wtests.txt
./nbody_cpu_multicore 28000 20 28 >> wtests.txt
./nbody_cpu_multicore 32000 20 32 >> wtests.txt
./nbody_cpu_multicore 44000 20 44 >> wtests.txt
./nbody_cpu_multicore 56000 20 56 >> wtests.txt
./nbody_cpu_multicore 100000 20 100 >> wtests.txt
