import numpy as np
# import Bio
# from Bio import PDB
# from Bio.PDB import PDBParser
# from Bio.PDB.vectors import Vector, rotmat
# from Bio.PDB import PDBIO
import os
import random
import math as m
import os
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime
import argparse

########################
# Scores all of the files in the sampled decoys with the following scoring functions:
# ZRANK, ITScore-Pro, DockQ, iAlign, Rosetta
########################

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", help="The name of the PDB you are testing")
parser.add_argument("--file_indicator", default="_H.pdb", help="Unique aspect of files you want to get")

def get_all_dockq_data_for_decoys(decoy_dir, dockq_dir, outfilename, native_path, start = 0, end = -1, save_dir="./", file_indicator = "_H.pdb"):
	'''
	Runs DockQ on all the decoys in decoy_dir and all the files in the dir collected in decoy_files.
	Ensure you specify the dockq_dir as well - use global path
	native_path is where the native structure is located - use global path
	save_dir is where the output file will be saved - use global path
	All data is stored in outfilename in the format:
	<decoy_filename.pdb>
	[All DockQ Output compared to native structure]
	'''
	print("BEGINNING DOCKGROUND RUNS")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and "ligand" not in f and "receptor" not in f])
	
	## Loop through all the files and run DockQ to store in the outfilename
	os.chdir(decoy_dir)
	outfilename = join(save_dir, outfilename) # Make sure it is saved in the right spot!
	new_file = open(outfilename, "w")

	if end == -1:
		end = len(all_decoys)

	for decoy in all_decoys[start:end]:
		
		## Run DockQ
		os.system(f"{dockq_dir}./DockQ.py {decoy} {native_path} > {decoy[:-4]}_temp.out")
		
		temp_file = open(f"{decoy[:-4]}_temp.out", "r")
		new_file.write(f"DockQ Data for {decoy}\n")
		for line in temp_file:
			if line[0] == "*":
				continue
			new_file.write(f"{line}")
		new_file.write("\n\n")
		temp_file.close()
		os.remove(f"{decoy[:-4]}_temp.out")
	new_file.close()
	print("DONE WITH DOCKGROUND RUNS")
	return 0

def get_all_ialign_scores(decoy_dir, ialign_dir, target_dir, pdb_name, filename_mod, chains = "AB", start = 0, end = -1, save_dir="./", i = 0, file_indicator = "_H.pdb"):
	'''
	Runs iAlign

	Copies the binary directory from the source code into the file and then removes it and the scratch folder
	when it is done.
	'''
	print("BEGINNING IALIGN SCORING RUNS")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f])

	if end == -1:
		end = len(all_decoys)

	## Create a quick file_list for ZRANK so it can run

	os.chdir(decoy_dir) ## go to decoy dir for simplicity
	temp_ialign_file = open(f"ialign_files_{filename_mod}.lst", "w")


	for file in all_decoys[start:end]:
		temp_ialign_file.write(file + f" {chains}\n")

	temp_ialign_file.close()

	## Run iAlign

	# copy ialign bin into directory
	os.system(f"cp -r {ialign_dir}bin {decoy_dir}")

	os.system(f"bin/isscore.pl -w scratch -l ialign_files_{filename_mod}.lst {target_dir} {chains} > {pdb_name}_{filename_mod}_iAlign_scores.txt")

	# remove zrank files from dir
	os.system("rm -r bin")
	os.system("rm -r scratch")

	## Move the ZRANK data to the 'save_dir'

	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)
	os.system(f"cp {pdb_name}_{filename_mod}_iAlign_scores.txt {save_dir}")


	print("DONE WITH IALIGN RUNS")
	return 0
	
def reformat_coords(filename):
	
	filename = filename[:-4]
	new_name = filename + "_mod.pdb"
	output = open(new_name, "w")

	with open(filename + ".pdb") as input:
		
		for line in input:
			
			if line.startswith("ATOM"):
			
				pdb_f_start = line[:30]
				
				x_coord = line[30:38]
				
				y_coord = line[38:46]
				
				z_coord = line[46:54]
							
				new_data = pdb_f_start + x_coord + " " + y_coord + " " + z_coord
				
				output.write(new_data + "\n")

	output.close()
	return new_name

def get_all_zdock_hdock_scores(decoy_dir, zrank_dir, hdock_dir, pdb_name, filename_mod, start = 0, end = -1, save_dir="./", i = 0, file_indicator = "_H.pdb"):
	'''
	Runs ZDOCK and HDOCK 
	'''
	print("BEGINNING ZRANK/HDOCK SCORING RUNS")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f])

	if end == -1:
		end = len(all_decoys)

	## Create a quick file_list for ZRANK so it can run

	os.chdir(decoy_dir) ## go to decoy dir for simplicity
	temp_zrank_file = open(f"zrank_files_{filename_mod}_{pdb_name}.txt", "w")


	for file in all_decoys[start:end]:
		temp_zrank_file.write(file + "\n")

	temp_zrank_file.close()

	## Run ZRANK quickly with the data

	# copy zrank files into directory
	os.system(f"cp {zrank_dir}* {decoy_dir}")

	os.system(f"./zrank zrank_files_{filename_mod}_{pdb_name}.txt")

	# remove zrank files from dir
	os.system("rm zrank")
	os.system("rm zrank_mac")
	os.system("rm README")

	## Move the ZRANK data to the 'save_dir'

	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)
	os.system(f"mv zrank_files_{filename_mod}_{pdb_name}.txt.zr.out {save_dir}")

	## Run HDOCK score for each file

	for decoy in all_decoys[start:end]:
		new_decoy = reformat_coords(decoy)
		os.system(f"{hdock_dir}./ITScorePro {new_decoy} >> hdock_scores_{filename_mod}_{pdb_name}.txt")
		os.system(f"rm {new_decoy}")

	os.system(f"mv hdock_scores_{filename_mod}_{pdb_name}.txt {save_dir}")


	print("DONE WITH ZRANK/HDOCK RUNS")
	return 0

def get_voromqa_scores(decoy_dir, voromqa_dir, outfilename, start = 0, end = -1, save_dir="./", file_indicator = "_H.pdb"):
	'''
	Runs DockQ on all the decoys in decoy_dir and all the files in the dir collected in decoy_files.
	Ensure you specify the dockq_dir as well - use global path
	native_path is where the native structure is located - use global path
	save_dir is where the output file will be saved - use global path
	All data is stored in outfilename in the format:
	<decoy_filename.pdb>
	[All DockQ Output compared to native structure]
	'''
	print("BEGINNING VoroMQA RUNS")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and "ligand" not in f and "receptor" not in f])
	
	## Loop through all the files and run DockQ to store in the outfilename
	os.chdir(decoy_dir)
	outfilename = join(save_dir, outfilename) # Make sure it is saved in the right spot!
	new_file = open(outfilename, "w")

	if end == -1:
		end = len(all_decoys)

	for decoy in all_decoys[start:end]:

		## Run VoroMQA
		os.system(f"{voromqa_dir}./voronota-voromqa --score-inter-chain -i {decoy} >> temp_voro_score.txt")
		
		temp_file = open("temp_voro_score.txt", "r")
		for line in temp_file:
			if "voromqa_v1_score" in line: ## it is the header line
				continue
			new_file.write(f"{line}")
		temp_file.close()
		os.remove(f"temp_voro_score.txt")
	new_file.close()
	print("DONE WITH VoroMQA RUNS")
	return 0

def get_pydock_scores(decoy_dir, pydock_dir, outfilename, chains = "AB", start = 0, end = -1, save_dir="./", file_indicator = "_H.pdb"):
	'''
	Runs DockQ on all the decoys in decoy_dir and all the files in the dir collected in decoy_files.
	Ensure you specify the dockq_dir as well - use global path
	native_path is where the native structure is located - use global path
	save_dir is where the output file will be saved - use global path
	All data is stored in outfilename in the format:
	<decoy_filename.pdb>
	[All DockQ Output compared to native structure]
	'''
	print("BEGINNING PyDOCK RUNS")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and "ligand" not in f and "receptor" not in f])
	
	## Loop through all the files and run DockQ to store in the outfilename
	os.chdir(decoy_dir)
	outfilename = join(save_dir, outfilename) # Make sure it is saved in the right spot!
	new_file = open(outfilename, "w")

	if end == -1:
		end = len(all_decoys)

	for decoy in all_decoys[start:end]:

		## Create input file
		filename_noext = decoy[:-4]
		output = open(filename_noext + ".ini", "w")
		output.write('''[receptor]
pdb     = {filename}
mol     = {chain1}
newmol  = {chain1}

[ligand]
pdb     = {filename}
mol     = {chain2}
newmol  = {chain2}'''.format(filename = decoy, chain1 = chains[0], chain2 = chains[1]))

		output.close()

		## Run PyDock
		os.system(f"{pydock_dir}./pyDock3 {filename_noext} bindEy")
		
		## Aggregate output into one file
		with open(filename_noext + ".ene") as infile:
			for line in infile:
				if line == "        Conf         Ele      Desolv         VDW       Total        RANK":
					continue

				elif line == "------------------------------------------------------------------------":
					continue

				else:
					scores = line

		new_file.write(decoy + "\t" + scores)
	new_file.close()
	print("DONE WITH PyDOCK RUNS")
	return 0

def get_complex_chains(complex_pdb):
	'''
	Gets the chains from the given PDB file 
	provided that pdb file is a dimer, or has two
	chains
	'''
	start = True
	chain_1 = ''
	chain_2 = ''
	with open(complex_pdb, "r") as temp_pdb:
		for line in temp_pdb:
			if line[:4] in 'ATOM':
				chain = line[21:23].strip()
				if start:
					chain_1 = chain 
					start = False
				if chain in chain_1:
					continue
				elif chain not in chain_1:
					chain_2 = chain 
					break
	return chain_1 + chain_2


def main():

	## Parse arguments

	args = parser.parse_args()
	pdb_name = args.pdb
	file_indicator = args.file_indicator
	
	## Set up the dirs

	sample_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}/"
	random_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}/random_negatives/"

	ialign_dir = "/home/jas485/PPI_project_work/ialign/"
	dockq_dir = "/home/jas485/PPI_project_work/DockQ/"
	zrank_dir = "/home/jas485/PPI_project_work/zrank/"
	hdock_dir = "/home/jas485/PPI_project_work/ITScorePro/"
	voromqa_dir = "/gpfs/gibbs/pi/ohern/jas485/voromqa/"
	pydock_dir = "/gpfs/gibbs/pi/ohern/jas485/pydock/"
	target_dir = f"/home/jas485/pdb_targets/{pdb_name}_complex_H.pdb"

	chains = get_complex_chains(target_dir)

	## Set up the save dir

	save_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/scores_sampled_{pdb_name}"
	if not isdir(save_dir):
		os.system(f"mkdir {save_dir}")

	######
	## Sampled decoys
	#####

	filename_mod = "sampled"

	outfilename = f"dockq_scores_{pdb_name}_{filename_mod}.txt"

	## DockQ Run

	get_all_dockq_data_for_decoys(
		sample_dir, 
		dockq_dir, 
		outfilename, 
		target_dir, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		file_indicator = file_indicator
	)

	## iAlign Run

	get_all_ialign_scores(
		sample_dir, 
		ialign_dir, 
		target_dir, 
		pdb_name, 
		filename_mod, 
		chains = chains, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		i = 0, 
		file_indicator = file_indicator
	)

	##	ZRank HDock Run

	get_all_zdock_hdock_scores(
		sample_dir, 
		zrank_dir,
		hdock_dir, 
		pdb_name, 
		filename_mod, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		i = 0, 
		file_indicator = file_indicator
	)

	## VoroMQA Run

	outfilename = f"voromqa_scores_{pdb_name}_{filename_mod}.txt"

	get_voromqa_scores(
		decoy_dir = sample_dir, 
		voromqa_dir = voromqa_dir, 
		outfilename = outfilename, 
		start = 0, 
		end = -1, 
		save_dir = save_dir, 
		file_indicator = file_indicator
	)

	## PyDock Run

	outfilename = f"pydock_scores_{pdb_name}_{filename_mod}.txt"

	get_pydock_scores(
		decoy_dir = sample_dir, 
		pydock_dir = pydock_dir, 
		outfilename = outfilename, 
		chains = chains, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		file_indicator = file_indicator
	)


	######
	## Random decoys
	#####

	filename_mod = "random"

	outfilename = f"dockq_scores_{pdb_name}_{filename_mod}.txt"

	## DockQ Run

	get_all_dockq_data_for_decoys(
		random_dir, 
		dockq_dir, 
		outfilename, 
		target_dir, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		file_indicator = file_indicator
	)

	## iAlign Run

	get_all_ialign_scores(
		random_dir, 
		ialign_dir, 
		target_dir, 
		pdb_name, 
		filename_mod, 
		chains = chains, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		i = 0, 
		file_indicator = file_indicator
	)

	##	ZRank HDock Run

	get_all_zdock_hdock_scores(
		random_dir, 
		zrank_dir,
		hdock_dir, 
		pdb_name, 
		filename_mod, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		i = 0, 
		file_indicator = file_indicator
	)

	## VoroMQA Run

	outfilename = f"voromqa_scores_{pdb_name}_{filename_mod}.txt"

	get_voromqa_scores(
		decoy_dir = random_dir, 
		voromqa_dir = voromqa_dir, 
		outfilename = outfilename, 
		start = 0, 
		end = -1, 
		save_dir = save_dir, 
		file_indicator = file_indicator
	)

	## PyDock Run

	outfilename = f"pydock_scores_{pdb_name}_{filename_mod}.txt"

	get_pydock_scores(
		decoy_dir = random_dir, 
		pydock_dir = pydock_dir, 
		outfilename = outfilename, 
		chains = chains, 
		start = 0, 
		end = -1, 
		save_dir=save_dir, 
		file_indicator = file_indicator
	)

	## Copy Rosetta Scores over

	sampled_ros_save = join(save_dir, f"rosetta_scores_sampled_{pdb_name}.txt")
	os.system(f"cp {sample_dir}score.sc {sampled_ros_save}")

	random_ros_save = join(save_dir, f"rosetta_scores_random_{pdb_name}.txt")
	os.system(f"cp {random_dir}score.sc {random_ros_save}")


if __name__ == '__main__':
	main()





