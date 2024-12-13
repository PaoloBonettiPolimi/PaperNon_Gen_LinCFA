#!/usr/bin/env bash

#SBATCH -J run_mnist_class_kerelPCA
#SBATCH --output run_mnist_class_kerelPCA.log
#SBATCH -p compute
#SBATCH -A bk1318
#SBATCH --time=08:00:00

source activate tensorflow_env # testEnv # tensorflowGpu_env
python run_mnist.py 
