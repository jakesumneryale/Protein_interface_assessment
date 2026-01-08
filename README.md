This is the code for the manuscript: "Assessment of scoring functions for computational models of protein-protein interfaces" By Jacob Sumner and Naomi Brandt et al. submitted in December 2025. 

The folders in this repository contain the following scripts and code: 

"uniform_sampling_cluster_scripts"

This folder contains all the scripts used on the cluster to run the uniform sampling code for each of the targets chosen. 

The models were all initially generated using the "supersample_run_zdock_and_dockq.py", which generates 540,000 models of each target and scores each one with DockQ relative to the crystal structure. 

The models are then uniformly sampled using the script "filter_decoys_supersample_preprocess_score.py", which uniformly samples the 540,000 targets across DockQ, yielding approximately 2000 models in the end (~1000 uniformly sampled models and an additional 1000 randomly sampled models). The script also adds hydrogens using the reduce script and then relaxes the structures using Rosetta relax. 

The uniformly sampled models are then scored for ZRank2, ITScorePP, Rosetta, PyDock, and VoroMQA using the script "score_all_sampled_decoys.py"

Physical features were generated using the scripts "physical_scoring_contacts_flatness.py" and "relative_interface_area_enum.py", to calculate the contacts, interface separability, and relative interface surface area. 




"jupyter_notebook_analysis_scripts"

This folder contains the Jupyter notebook files that were used for data analysis and figure generation.

The large notebook: "PPI_manuscript_figures_and_code.ipynb" contains much of the original code for data analysis and processing. 

The more organized notebook: "cleaned_PPI_manuscript_code.ipynb" contains the code to generate all the figures for the manuscript. 
