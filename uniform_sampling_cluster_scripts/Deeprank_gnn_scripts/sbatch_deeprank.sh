#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=1acb_rand_deeprank
##SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-core=1
#SBATCH -c 2
#SBATCH -Q
#SBATCH --mem-per-cpu=25G
#SBATCH --gpus=1
#SBATCH --constraint="p100|v100|rtx2080ti|rtx5000"
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=naomi.brandt@yale.edu
#SBATCH --output=submit_1acb_rand.out
#SBATCH --requeue
##SBATCH --array=1-2

# This command navigates to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR


# This command sends the tasklist to all in the array
./run_deeprank_1acb.sh

