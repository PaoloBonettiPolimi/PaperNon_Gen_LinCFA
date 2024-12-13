#!/usr/bin/env bash

#SBATCH -J kernelPCA
#SBATCH --output fashionMnist_kernelPCA.log
#SBATCH -p compute
#SBATCH -A bk1318
#SBATCH --time=08:00:00

source activate tensorflow_env # testEnv # tensorflowGpu_env
python run_fashion_mnist.py 
