#!/bin/bash

module load miniconda
conda activate DeepRank_gpu

deeprank-gnn-esm-predict /gpfs/gibbs/pi/ohern/nb685/Decoys/Supersampled_structures/sampled_1acb/random_negatives/rand_1acb_relaxed/1acb_NoH/ 1acb
