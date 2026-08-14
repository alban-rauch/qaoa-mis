#!/bin/bash
################################################################################################
#SBATCH -J speed_test               # Job name               (or --job-name=quantum_sim)
#SBATCH -p MI350P_600W              # Partition name         (or --partition=0745-1R5600-NOIB)
#SBATCH -N 1                        # Nodes requested        (or --nodes=1)
#SBATCH --exclusive                 # Resource not shared with other users
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=00:05:00             # Time limit
#SBATCH --output=outputs/%x_%j.out  # Output file
#SBATCH --error=outputs/%x_%j.err   # Error file
################################################################################################

module purge
module load pennylane-amdgpu/0.45.0-rocm7.1.1-gfx950

cd ~/test

srun python -c "print('Hello')"