Naomi Brandt
8/27/24

This directory contains the following:


predict_savefasta.py - predict.py edited to use correct FASTA sequencing, run entire folder of models and save output in .csv format, not require chain ID input/editing -> run with original version of GraphGenMP.py

predict_reusefasta.py - same functionalities as predict_savefasta.py, also edited to now use the same FASTA and ESM embeddings for models generated from the same monomers (since both are determined by sequence, not structure), now requires pdbid or another identifier as argument to run main -> run with GraphGenMP_reusefasta.py

GraphGenMP_reusefasta.py - GraphGenMP.py edited to now use the same FASTA and ESM embedding - quality of life to not regenerate FASTA each run for models generated from the same pdb

DeepRank-GNN-ESM_env.yml - conda environment for running DeepRank-GNN-esm

run_deeprank.sh - executable for running DeepRank on folder of models

sbatch_deeprank.sh - sbatch command to put run_deeprank.sh to the ycrc Grace cluster
