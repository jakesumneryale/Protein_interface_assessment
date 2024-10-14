#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=dovescore_3k9p_f5
##SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-core=1
#SBATCH -c 2
#SBATCH -Q
#SBATCH --mem-per-cpu=25G
#SBATCH --gpus=1
#SBATCH --constraint="rtx5000|gtx1080ti"
#SBATCH -t 6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=naomi.brandt@yale.edu
#SBATCH --output=submit_3k9p_fold5.out
#SBATCH --requeue
##SBATCH --array=1-2

# This command navigates to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# This command sends the tasklist to all in the array
#module load miniconda
./run_dove_3k9p_f5.sh



