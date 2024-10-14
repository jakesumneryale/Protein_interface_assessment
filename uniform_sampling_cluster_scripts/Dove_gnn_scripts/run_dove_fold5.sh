#!/bin/bash

CONDA_ENVS_PATH=/gpfs/gibbs/pi/ohern/nb685/conda_envs

module load miniconda
conda activate GNN_DOVE


Targ_path=/gpfs/gibbs/pi/ohern/nb685/Decoys//Non_SS/3k9p/3k9p_NoH
python main.py --mode=1 -F ${Targ_path} --gpu=${CUDA_VISIBLE_DEVICES} --fold=5 --receptor_units=1



