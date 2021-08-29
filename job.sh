#!/bin/bash
module load cuda/10.0
module load papi
module load intel
#sinteractive --account=mpcs52018 --exclusive --partition=gpu2 --nodes=1 --time=00:10:00
sinteractive --exclusive --partition=gpu2 --nodes=1 --gres=gpu:1 --account=mpcs52018
