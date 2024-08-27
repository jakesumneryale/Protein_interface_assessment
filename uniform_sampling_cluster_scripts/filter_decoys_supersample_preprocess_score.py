import numpy as np
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.vectors import Vector, rotmat
from Bio.PDB import PDBIO
import matplotlib.pyplot as plt
import os
import random
import math as m
import os
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime
import argparse
import pandas as pd

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--n_runs", help="Number of ZDOCK runs")
parser.add_argument("--pdb", help="The 4 character pdb ID with only lowercase letters")

def create_dockq_df_multi_file(dockq_dir, hdock = False):
	'''
	Creates a pandas dataframe from the dockq data for each
	decoy.
	It returns a dictionary of dataframes, with the key being the pdb ID
	'''
	os.chdir(dockq_dir)
	temp_dict = {}
	dockq_files = sorted([f for f in listdir(dockq_dir) if isfile(join(dockq_dir, f)) and ".txt" in f])
	
	## Parses through the files, storing the data in a pandas dataframe
	count = 0
	for score_file in dockq_files:
		temp_decoy_dockq = get_dockq_results_from_output(dockq_dir, score_file)
		
		for decoy, scores in list(temp_decoy_dockq.items()):
			if "Decoy" not in temp_dict:
				temp_dict["Decoy"] = []
			if "Model" not in temp_dict:
				temp_dict["Model"] = []
			if "Fnat" not in temp_dict:
				temp_dict["Fnat"] = []
			if "iRMSD" not in temp_dict:
				temp_dict["iRMSD"] = []
			if "LRMSD" not in temp_dict:
				temp_dict["LRMSD"] = []
			if "CAPRI" not in temp_dict:
				temp_dict["CAPRI"] = []
			if "DockQ" not in temp_dict:
				temp_dict["DockQ"] = []
			temp_dict["Decoy"].append(decoy.split("_corrected_H")[0])   # index
			if hdock:
				temp_dict["Model"].append(int(decoy.split(".")[1].split("_")[0]))
			else:
				temp_dict["Model"].append(int(decoy.split(".")[1].split("_")[0]))
			temp_dict["Fnat"].append(scores[0]) # iAlign score
			temp_dict["iRMSD"].append(scores[1]) # iRMSD
			temp_dict["LRMSD"].append(scores[2]) # norm RMSD
			temp_dict["CAPRI"].append(scores[3]) # ZRank score
			temp_dict["DockQ"].append(scores[4]) # HDock score
			count += 1
	print("DOCKQ COUNT:", count)
	
	return pd.DataFrame(temp_dict)

def get_dockq_results_from_output(dir_loc, dockq_filename):
	'''
	take the dir_loc and dockq_filename and pulls all the data for each file
	and puts it in a dictionary, which is returned
	'''
	final_dict = {}
	os.chdir(dir_loc)
	file_obj = open(dockq_filename, "r")
	temp_name = ''
	temp_lst = []
	for line in file_obj:
		if line[0] == " ":
			continue
		elif line[:10] in "DockQ Data":
			temp_name = line.split()[-1]
		elif line[:4] in "Fnat":
			fnat = float(line.split()[1])
			temp_lst.append(fnat)
		elif line[:4] in "iRMS":
			irms = float(line.split()[1])
			temp_lst.append(irms)
		elif line[:4] in "LRMS":
			lrms = float(line.split()[1])
			temp_lst.append(lrms)
		elif line[:5] in "CAPRI":
			capri = line.split()[1]
			temp_lst.append(capri)
		elif line[:5] in "DockQ":
			## Final entry for the element -> add to dictionary and reset
			dockq = float(line.split()[1])
			temp_lst.append(dockq)
			final_dict[temp_name] = temp_lst
			temp_name = ''
			temp_lst = []
	return final_dict

def get_zrank_score(zrank_dir, hdock = False):
	os.chdir(zrank_dir)
	
	zrank_files = sorted([f for f in listdir(zrank_dir) if isfile(join(zrank_dir, f)) and ".zr.out" in f])
	
	zrank_data = {}
	
	count = 0
	for score_file in zrank_files:
		temp_obj = open(score_file, "r")
		for line in temp_obj:
			line = line.split()
			score = float(line[1])
			if hdock:
				decoy = line[0].split("_corrected_H")[0]
			else:
				decoy = line[0].split("_corrected_H")[0]
			if "Decoy" not in zrank_data:
				zrank_data["Decoy"] = []
			if "ZRank" not in zrank_data:
				zrank_data["ZRank"] = []
			
			zrank_data["Decoy"].append(decoy)
			zrank_data["ZRank"].append(score)
			count += 1
	
	print("ZRANK COUNT:", count)
	return pd.DataFrame(zrank_data)

def get_hdock_score(hdock_dir, hdock = False):
	os.chdir(hdock_dir)
	
	hdock_files = sorted([f for f in listdir(hdock_dir) if isfile(join(hdock_dir, f)) and "hdock_scores" in f])
	
	hdock_data = {}
	
	count = 0
	for score_file in hdock_files:
		temp_obj = open(score_file, "r")
		for line in temp_obj:
			line = line.split()
			score = float(line[3])
			if hdock:
				decoy = line[0].split("_corrected_H")[0]
			else:
				decoy = line[0].split("_corrected_H")[0]
			if "Decoy" not in hdock_data:
				hdock_data["Decoy"] = []
			if "HDOCK" not in hdock_data:
				hdock_data["HDOCK"] = []
			if decoy not in hdock_data["Decoy"]:
				hdock_data["Decoy"].append(decoy)
				hdock_data["HDOCK"].append(score)
			count += 1
	print("HDOCK COUNT:", count)
	return pd.DataFrame(hdock_data)

def z_score_normalize_df_column(df, ref_column_name, new_column_name):
	'''
	Normalizes the data in "ref_column_name" and adds the normalized data 
	to 'new_column_name'. No return necessary, since df is an existing object
	which is modified
	'''
	df[new_column_name] = (df[ref_column_name] - np.mean(df[ref_column_name]))/np.std(df[ref_column_name])
	return df

def get_list_of_top_decoys(dockq_dir):
	'''
	Returns the top 1000 and random 500 decoys
	sorted by the DockQ data
	'''
	temp_df = create_dockq_df_multi_file(dockq_dir)
	temp_df = temp_df.sort_values(by = ["DockQ"], ascending = False)
	top_1000_decoy = list(temp_df["Decoy"])[:1000]
	top_1000_dockq = list(temp_df["DockQ"][:1000])
	
	top_1000 = [(ele, top_1000_dockq[i]) for i,ele in enumerate(top_1000_decoy)]
	
	rand_500_decoy = list(temp_df["Decoy"])[1000:]
	rand_500_dockq = list(temp_df["DockQ"])[1000:]
	rest_shuffled = [(ele, rand_500_dockq[i]) for i, ele in enumerate(rand_500_decoy)]
	random.shuffle(rest_shuffled)
	rand_1000 = rest_shuffled[:1000]
		
	return (top_1000, rand_1000)

def get_top_decoys_from_dockq_data(n_runs, dockq_dir_spec):
	'''
	Gets all the top/random decoys from the dockq
	data for each run of ZDOCK
	'''
	decoy_list = []
	
	for i in range(n_runs):
		dockq_dir = dockq_dir_spec.format(run = i)
		decoy_list.append(get_list_of_top_decoys(dockq_dir))
	
	return decoy_list

def get_top_decoys_from_dockq_data_2(n_runs, dockq_dir_spec):
	'''
	Gets all the top/random decoys from the dockq
	data for each run of ZDOCK
	'''
	decoy_list = []
	
	for i in range(n_runs):
		dockq_dir = dockq_dir_spec.format(run = i+1)
		decoy_list.append(get_list_of_top_decoys(dockq_dir))
	
	return decoy_list

def rename_combine_sample_decoys(decoy_list, n_bins = 20, n_samples = 50):
	'''
	Renames all decoys,
	combines them all into a list
	separates the dockq scores into separate lists according n_bins
	using n_samples number of decoys
	'''
	combined_decoys = []
	combined_dockq = []
	rand_only = []
	for i, (top_1000, rand_1000) in enumerate(decoy_list):
		for decoy, dockq in top_1000:
			decoy = decoy.split(".")
			new_decoy = f"{decoy[0]}.{decoy[1]}.{i}.pdb" ## includes the run number - i
			combined_decoys.append(new_decoy)
			combined_dockq.append(dockq)
		for decoy, dockq in rand_1000:
			decoy = decoy.split(".")
			new_decoy = f"{decoy[0]}.{decoy[1]}.{i}.pdb" ## includes the run number - i
			combined_decoys.append(new_decoy)
			rand_only.append(new_decoy)
			combined_dockq.append(dockq)
			
			
	combined_decoys = np.array(combined_decoys)
	combined_dockq = np.array(combined_dockq)
			
	## Separate the combined decoys into a list of lists according to the bins
	
	binned_data = []
	bin_cutoffs = np.linspace(0,1, n_bins+1)
	for i in range(n_bins):
		min_bin = bin_cutoffs[i]
		max_bin = bin_cutoffs[i+1]
		
		## Gets the decoys only in that specific DockQ score bin
		temp_data = list(combined_decoys[(combined_dockq > min_bin) & (combined_dockq <= max_bin)])
		random.shuffle(temp_data)
		binned_data.append(temp_data[:n_samples])
		
	random.shuffle(rand_only)
	final_rand_1000 = rand_only[:1000]
	return binned_data, final_rand_1000
			

def rename_and_move_decoys(parent_dir, decoy_list, mv_dir):
	'''
	Renames all the files in the decoy list according to
	their run number and moves them to a new directory
	'''
	binned_data, rand_negs = rename_combine_sample_decoys(decoy_list, n_bins = 20, n_samples = 50)
	
	## Loop through the binned data to move the files accordingly
	for i,n_bin in enumerate(binned_data):
		for j,decoy in enumerate(n_bin):
			
			## Normal run for many runs
			split_decoy = decoy.split(".")
			run_num = int(split_decoy[2])
			orig_decoy = ".".join(split_decoy[:2] + split_decoy[3:])
			new_decoy_name = f"complex.{run_num}_{i}_{j}.pdb"
			os.chdir(parent_dir.format(run = run_num)) 
			
			## ensure directory exists - if not create
			if not isdir(mv_dir):
				os.system(f"mkdir {mv_dir}")
				
			comb_file_and_dir = join(mv_dir, new_decoy_name)
			os.system(f"cp {orig_decoy} {comb_file_and_dir}")
			
	for decoy in rand_negs:
		split_decoy = decoy.split(".")
		run_num = int(split_decoy[2])
		orig_decoy = ".".join(split_decoy[:2] + split_decoy[3:])
		new_decoy_name = f"complex.{split_decoy[1]}_{run_num}.pdb"
		os.chdir(parent_dir.format(run = run_num))
		neg_dir = join(mv_dir,"random_negatives")
		
		## Create directory if it doesn't already exist
		if not isdir(neg_dir):
			
			os.system(f"mkdir {neg_dir}")
		comb_file_and_dir = join(neg_dir, new_decoy_name)
		os.system(f"cp {orig_decoy} {comb_file_and_dir}")
	
	## Returns the two directories where the sampled files are located for easy preprocessing
	return mv_dir, neg_dir


def create_cleanup_taskfile(base_dir, pdb_name):
	'''
	Creates a taskfile where each line calls the
	cleanup_supersampling.py script so that the cleanup
	process can happen in parallel
	'''

	taskfile_name = "cleanup_task.sh"

	zdock_dirs = sorted([f for f in listdir(base_dir) if isdir(join(base_dir, f)) and f"{pdb_name}_" in f])

	with open(taskfile_name, "w") as f:
		for i,z_dir in enumerate(zdock_dirs):
			parent_dir = join(base_dir, z_dir)
			complete_dir = f"{pdb_name}_{i}_zdock/"
			f.write(f"python cleanup_supersampling.py --parent_dir {parent_dir} --dir_to_clean {complete_dir}\n")

	f.close()
	return len(zdock_dirs)


def modify_job_script(base_dir, num_jobs):
	'''
	After the job script has been copied over, this job will
	add the number of jobs needed to be run to the array command
	if it is not 10 (default)
	'''

	#SBATCH --array=1-66

	job_file = open(join(base_dir, "clean_job.sh"), "r")
	with open("temp.sh", "w") as f:
		for line in job_file:
			if "array" in line:
				f.write(f"#SBATCH --array=1-{num_jobs}")
			else:
				f.write(line)

	f.close()
	return 0

def create_rosetta_relax_taskfile(sample_dir, negative_dir, num_per_task, file_indicator = "_H.pdb"):
	'''
	Creates the rosetta relax taskfile to run the decoys in the 
	preprocessed directories. The num_per_task parameter
	defines the number of runs that we will bunch into one run
	'''

	taskfile_name = "relax_task.sh"

	sample_decoys = sorted([f for f in listdir(sample_dir) if isfile(join(sample_dir, f)) and file_indicator in f])
	negative_decoys = sorted([f for f in listdir(negative_dir) if isfile(join(negative_dir, f)) and file_indicator in f])

	with open(taskfile_name, "w") as f:
		for i in range(0, len(sample_decoys), num_per_task):
			start = i
			end = i + num_per_task
			f.write(f"python run_rosetta_relax_enum.py --parent_dir {sample_dir} --start {start} --end {end}\n")
		for i in range(0, len(negative_decoys), num_per_task):
			start = i
			end = i + num_per_task
			f.write(f"python run_rosetta_relax_enum.py --parent_dir {negative_dir} --start {start} --end {end}\n")
	f.close()

	return 0


def main():

	## Parse through args

	args = parser.parse_args()

	n_runs = int(args.n_runs)
	pdb_name = args.pdb

	## Create a base dir for easy reference

	base_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/"

	## Get the top decoys from each folder

	dockq_dir_str = f"{pdb_name}_" + "{run}/" + f"{pdb_name}_" + "{run}_zdock/dockq_output"

	dockq_dir = join(base_dir,dockq_dir_str)

	decoy_list = get_top_decoys_from_dockq_data(n_runs, dockq_dir)

	## Sample and copy files to new directories

	zdock_dir_str = f"{pdb_name}_" + "{run}/" + f"{pdb_name}_" + "{run}_zdock"

	parent_dir = join(base_dir, zdock_dir_str)
	mv_dir = join(base_dir, f"sampled_{pdb_name}")

	mv_dir, neg_dir = rename_and_move_decoys(
						parent_dir,
						decoy_list,
						mv_dir
					)

	## Run preprocess on the decoys - sampled decoys

	num_files = len([f for f in listdir(mv_dir) if isfile(join(mv_dir, f)) and ".pdb" in f]) ## there may not be precisely 1000 files, so get the actual number
	os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/preprocess_source_files/* {mv_dir}")
	os.chdir(mv_dir)
	os.system(f"python preprocess.py -n {num_files}")
	os.system("rm *corrected.txt")

	## Run preprocess on the negative decoys

	os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/preprocess_source_files/* {neg_dir}")
	os.chdir(neg_dir)
	os.system("python preprocess.py -n 1000")
	os.system("rm *corrected.txt")

	## Clean up the files in parallel by creating a job and submitting it

	os.chdir(base_dir)
	num_dirs = create_cleanup_taskfile(base_dir, pdb_name) ## Creates the taskfile
	os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/cleanup_scripts/clean_job.sh {base_dir}")
	if num_dirs != 10: ## default is 10 jobs
		modify_job_script(base_dir, num_dirs)
	os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/cleanup_scripts/cleanup_supersampling.py {base_dir}")
	os.system(f"sbatch clean_job.sh")

	## Run Rosetta Relax on the decoys in the two directories

	os.chdir(base_dir)

	create_rosetta_relax_taskfile(
		mv_dir,
		neg_dir,
		3
		)

	os.system("cp /gpfs/gibbs/pi/ohern/jas485/rosetta_relax_scripts/submit_relax.sh ./")
	os.system("cp /gpfs/gibbs/pi/ohern/jas485/rosetta_relax_scripts/run_rosetta_relax_enum.py ./")

	os.system("sbatch submit_relax.sh")

	## Score the relaxed files?? Fuck this is a long script






if __name__ == '__main__':
	main()






