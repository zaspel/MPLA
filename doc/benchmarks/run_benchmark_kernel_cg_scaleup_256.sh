#!/bin/bash -l
#SBATCH --job-name=benchmark_kernel_cg_256
#SBATCH --time=00:10:00
#SBATCH --nodes=256
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=results_%x.%j.o

module load daint-gpu
module load cudatoolkit
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/pzaspel/MPLA/src:.
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$MPICH_DIR/lib/pkgconfig
export MPICH_RDMA_ENABLED_CUDA=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./benchmark_kernel_cg $[2**21] 5
