#!/bin/bash
#SBATCH --job-name=download-GOES
#SBATCH --output=./output/download-%j.out
#SBATCH --error=./output/download-%j.err
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --tasks-per-node=1

# Load Python 3.8.3
module load Python/3.8.3

# Run Python script
srun ./download_raw_data.py
