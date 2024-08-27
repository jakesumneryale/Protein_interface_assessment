import numpy as np
import Bio
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
import scipy as sp
import os
import random
import math as m
import os
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime
import argparse
import glob
from sklearn import svm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

########################
# Gets all of the contacts and flatness scores for each decoy in a
# given set
########################

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", help="The name of the PDB you are testing")
parser.add_argument("--file_indicator", default="_H.pdb", help="Unique aspect of files you want to get")
parser.add_argument("--exhaustive", default = False, action = "store_true", help="Indicates that you are running exhaustive data")
parser.add_argument("--i", default=0, type=int, help="Run number of the exhaustive decoys")

## Get c-alpha and heavy atom coordinates for each structure

def get_atomic_coords(decoy):
	'''
	Decoy is the input decoy filename with extension
	Returns the atomic coordinates for both chains.
	For each chain, two sets of coordinates are returned, 
	the heavy atom coordinates and the c-alpha only coordiantes.
	Additionally, indexes for each atom are in a separate array
	so it is obvious which residue identity each atom belongs to.
	All are returned in numpy arrays. 
	'''

	p = PDBParser()
	decoy_name = decoy.split(".pdb")
	structure = p.get_structure(decoy_name, decoy) 
	
	c_alpha_rec_list = []
	c_alpha_rec_resindex = []
	c_alpha_lig_list = []
	c_alpha_lig_resindex = []
	heavy_atom_rec_list = []
	heavy_atom_rec_resindex = []
	heavy_atom_lig_list = []
	heavy_atom_lig_resindex = []

	first_chain = True
	for chains in structure:
		for chain in chains:
			for residue in chain:  
				## Gets the residue number in the context of the PDB
				for atom in residue:
					## First chain is the 'receptor'
					if first_chain:
						atom_name = atom.get_name()
						residue_num = atom.get_parent()
						if "CA" in atom_name:
							c_alpha_rec_list.append(np.array(atom.get_coord()))
							c_alpha_rec_resindex.append(residue_num.id[1])
						if (atom_name[0] != 'H' and (atom_name[0] not in [str(0), str(1), str(2), str(3)])) and 'UNK' not in atom_name:
							heavy_atom_rec_list.append(np.array(atom.get_coord()))
							heavy_atom_rec_resindex.append(residue_num.id[1])
					
					else:
						atom_name = atom.get_name()
						residue_num = atom.get_parent()
						if "CA" in atom_name:
							c_alpha_lig_list.append(np.array(atom.get_coord()))
							c_alpha_lig_resindex.append(residue_num.id[1])
						if (atom_name[0] != 'H' and (atom_name[0] not in [str(0), str(1), str(2), str(3)])) and 'UNK' not in atom_name:
							heavy_atom_lig_list.append(np.array(atom.get_coord()))
							heavy_atom_lig_resindex.append(residue_num.id[1])
						
			## first chain is done
			first_chain = False
	
	
	c_alpha_rec_list = np.array(c_alpha_rec_list)
	c_alpha_rec_resindex = np.array(c_alpha_rec_resindex)
	c_alpha_lig_list = np.array(c_alpha_lig_list)
	c_alpha_lig_resindex = np.array(c_alpha_lig_resindex)
	heavy_atom_rec_list = np.array(heavy_atom_rec_list)
	heavy_atom_rec_resindex = np.array(heavy_atom_rec_resindex)
	heavy_atom_lig_list = np.array(heavy_atom_lig_list)
	heavy_atom_lig_resindex = np.array(heavy_atom_lig_resindex)
	
	return c_alpha_rec_list, c_alpha_lig_list, c_alpha_rec_resindex, c_alpha_lig_resindex, heavy_atom_rec_list, heavy_atom_lig_list, heavy_atom_rec_resindex, heavy_atom_lig_resindex
					

## Code for determining what is or isn't an interfacial contact

def get_c_alpha_contacts(ca_rec, ca_lig):
	'''
	Returns the number of C-alpha contacts between the
	protein receptor and ligand pairs that are provided.
	Returns both the distances within 10Å and 8Å
	'''
	dist_mat = sp.spatial.distance_matrix(ca_rec, ca_lig)
	
	flat_mat = dist_mat.flatten()
	
	contacts_arr = []
	for cutoff in np.linspace(1,30, 291):
		contacts_arr.append(len(flat_mat[flat_mat<=cutoff])) ## Get the cutoff
	
	return contacts_arr

def get_heavy_atom_contacts(ha_rec, ha_lig, ha_rec_ind, ha_lig_ind):
	'''
	Returns the heavy atom contacts between the
	protein receptor and ligand pairs that are provided.
	The distance cutoff between the heavy atoms is set to 
	5Å, which is the value used in DockQ. Only one contact 
	total can exist within a given residue pair
	'''
	
	## Get all the contacts and then loop through afterwards to check for repeats
	
	contact_pair_set = set([])
	
	dist_mat = sp.spatial.distance_matrix(ha_rec, ha_lig)

	flat_mat = dist_mat.flatten()
	
	contacts_arr = []
	for cutoff in np.linspace(1,30, 291):
		contacts_arr.append(len(flat_mat[flat_mat<=cutoff])) ## Get the cutoff
	
	return contacts_arr

def get_all_contacts(decoy_dir, outfilename, save_dir = "./", file_indicator = "_H.pdb"):
	'''
	Loops through the decoy dir and gets all the files with the given
	file_indicator. Gets the relevant C-alpha and heavy atom contacts
	and saves them to a .csv file with the relevant outfilename
	'''

	os.chdir(decoy_dir)

	## Define the names for the distances
	ca_names_arr = []
	name_start = "ca_"
	for cutoff in np.linspace(1,30, 291):
		ca_names_arr.append(name_start+str(cutoff)) 

	ha_names_arr = []
	name_start = "ha_"
	for cutoff in np.linspace(1,30, 291):
		ha_names_arr.append(name_start+str(cutoff)) 
	

	## Initialize the dictionary

	decoy_dict = {"decoy" : []}

	for i in range(291):
		decoy_dict[ca_names_arr[i]] = []

	for i in range(291):
		decoy_dict[ha_names_arr[i]] = []
	
	## Get all of the decoys 
	
	print("BEGINNING Contact Counting")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and "ligand" not in f and "receptor" not in f and f.endswith(".pdb") and "lig.pdb" not in f and "rec.pdb" not in f])
		
	for decoy in all_decoys:
		
		decoy_name = decoy[:-4]

		decoy_dict["decoy"].append(decoy_name)
		
		## Get the coordinates and relevant arrays for the decoy
		ca_rec, ca_lig, ca_ind_rec, ca_ind_lig, ha_rec, ha_lig, ha_ind_rec, ha_ind_lig = get_atomic_coords(decoy)
		
		## Get C-alpha contacts
		ca_contacts = get_c_alpha_contacts(ca_rec, ca_lig)

		for i,contact in enumerate(ca_contacts):
			decoy_dict[ca_names_arr[i]].append(contact)
		
		## Get heavy atom contacts - 5 Å
		
		ha_contacts = get_heavy_atom_contacts(ha_rec, ha_lig, ha_ind_rec, ha_ind_lig)
	
		for i,contact in enumerate(ha_contacts):
			decoy_dict[ha_names_arr[i]].append(contact)
		
	temp_df = pd.DataFrame(decoy_dict)
	
	## Save the data
	
	os.chdir(save_dir)
	
	temp_df.to_csv(outfilename)
	
	return temp_df


def split_pdb_chains(decoy, decoy_dir = "./"):
	'''
	Splits the PDB up into its individual substituent chains.
	labels them currently as temp_chain_1.pdb and temp_chain_2.pdb
	for simplicity. Will save these structures in the 
	directory where the decoys are located so there aren't any 
	complications with file accesses when running in parallel. 
	'''
	parser = PDBParser()
	io = PDBIO()

	os.chdir(decoy_dir)

	structure = parser.get_structure(decoy[:-4], decoy)
	pdb_chains = structure.get_chains()
	num = 1
	for chain in pdb_chains:
		io.set_structure(chain)
		io.save(f"temp_chain_{num}.pdb")
		num += 1
		
	return 0

def get_decoy_intertwined_values(decoy_dir, outfilename, save_dir, file_indicator = "_H.pdb"):
	'''
	Get the decoy intertwined values for all the decoys present in
	the decoy dir with the given file indicator
	'''

	p = PDBParser()

	os.chdir(decoy_dir)

	print("BEGINNING Flatness Score")
	all_decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and "ligand" not in f and "receptor" not in f and f.endswith(".pdb") and "lig.pdb" not in f and "rec.pdb" not in f])

	decoy_names = []

	acc_list = []
	for decoy in all_decoys:

		## Save name

		decoy_names.append(decoy[:-4])

		## Load pdb into parser obj
		
		coords = []
		label_list = []
		
		## Split up decoy into two chains

		split_pdb_chains(decoy, decoy_dir = decoy_dir)

		## Collect coordinates for each

		structure = p.get_structure("temp1", "temp_chain_1.pdb") 
		for chains in structure:
			for chain in chains:
				for residue in chain:  
					for atom in residue:
						coords.append(list(atom.get_coord()))
						label_list.append(0)
		
		structure = p.get_structure("temp2", "temp_chain_2.pdb") 
		for chains in structure:
			for chain in chains:
				for residue in chain:  
					for atom in residue:
						coords.append(list(atom.get_coord()))
						label_list.append(1)
					
		clf = svm.SVC(kernel='poly',degree=3,max_iter=1000000,verbose=0)
		clf.fit(coords, label_list)
		
		acc = clf.score(coords, label_list)
		acc_list.append(acc)
		
		params = clf.get_params(deep=True)

	#%%

	with open('poly3_SVM_acc_targets.txt','w') as f:
		
		for i in range(0,len(decoy_names)):
			
			f.write(decoy_names[i]+'\t '+str(acc_list[i]))
			f.write('\n')

	temp_dict = {"decoy" : decoy_names,
				 "flatness_p3" : acc_list}


	## Save the data as a CSV
	os.chdir(save_dir)

	temp_df = pd.DataFrame(temp_dict)

	temp_df.to_csv(outfilename)

	return temp_dict

def main():
	### Run the script

	## Parse arguments

	args = parser.parse_args()
	pdb_name = args.pdb
	file_indicator = args.file_indicator
	exhaustive_bool = args.exhaustive
	run_num = args.i
	
	if exhaustive_bool:
		
		## This will be the exhastive data
		
		sample_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/exhaustive_{pdb_name}/{pdb_name}_{run_num}_exhaustive/"
		
		save_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/exhaustive_{pdb_name}/exhaustive_scores_{pdb_name}"
		if not isdir(save_dir):
			os.system(f"mkdir {save_dir}")
			
		filename_mod = "exhaustive"
		
		outfilename = f"contacts_{pdb_name}_{filename_mod}_{run_num}.txt"
	
		## Contact Run
	
		get_all_contacts(
			decoy_dir = sample_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)
	
		## Intertwined Run
	
		outfilename = f"intertwined_{pdb_name}_{filename_mod}_{run_num}.txt"
	
		get_decoy_intertwined_values(
			decoy_dir = sample_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)
		
	#######################################
	##### ORIGINAL SUPERSAMPLING CODE #####
	#######################################
	
	elif not exhaustive_bool:
	
		## Set up the dirs
	
		sample_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}/"
		random_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}/random_negatives/"
	
	
		save_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/scores_sampled_{pdb_name}"
		if not isdir(save_dir):
			os.system(f"mkdir {save_dir}")
	
		######
		## Sampled decoys
		######
	
		filename_mod = "sampled"
	
		outfilename = f"contacts_{pdb_name}_{filename_mod}.txt"
	
		## Contact Run
	
		get_all_contacts(
			decoy_dir = sample_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)
	
		## Intertwined Run
	
		outfilename = f"intertwined_{pdb_name}_{filename_mod}.txt"
	
		get_decoy_intertwined_values(
			decoy_dir = sample_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)
	
		######
		## Random decoys
		######
	
		filename_mod = "random"
	
		outfilename = f"contacts_{pdb_name}_{filename_mod}.txt"
	
		## Contact Run
	
		get_all_contacts(
			decoy_dir = random_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)
	
		## Intertwined Run
	
		outfilename = f"intertwined_{pdb_name}_{filename_mod}.txt"
	
		get_decoy_intertwined_values(
			decoy_dir = random_dir, 
			outfilename = outfilename, 
			save_dir = save_dir, 
			file_indicator = file_indicator
		)


if __name__ == '__main__':
	main()
	











	
