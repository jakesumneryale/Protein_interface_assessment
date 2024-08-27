import freesasa
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
import argparse

'''
Calculate the relative interface area of the complexes using FreeSASA
'''

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", help="The name of the PDB you are testing")
parser.add_argument("--file_indicator", default="_H.pdb", help="Unique aspect of files you want to get")
parser.add_argument("--exhaustive", default = False, action = "store_true", help="Indicates that you are running exhaustive data")
parser.add_argument("--i", default=0, type=int, help="Run number of the exhaustive decoys")

def calculate_sasa_many(decoy_dir, file_indicator, freesasa_params = freesasa.Parameters()):
	'''
	Calculates the SASA for all decoys in the decoy dir that have the 
	file_indicator string in their filename
	'''

	os.chdir(decoy_dir)
	decoys = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and file_indicator in f and f[-4:] in ".pdb"])

	## Read in Hydrogens
	freesasa.Structure.defaultOptions["hydrogen"] = True

	decoy_names = []
	decoy_sasas = []

	for decoy in decoys:
		structure = freesasa.Structure(decoy)
		result = freesasa.calc(structure, freesasa_params)
		decoy_sasas.append(result.totalArea())
		decoy_names.append(decoy.split(".pdb")[0])

	return decoy_names, decoy_sasas

def write_output_file(new_filename, decoy_names, decoy_sasas, total_monomer_area, dir_loc = "/"):
	'''
	Writes the output file with the name of the new_filename provided.
	The decoy name from decoy_names is in the first column and the
	corresponding decoy SASA from the decoy_sasas argument is in the second
	column of the comma separated file. Total_monomer_area is added
	in the 3rd column as a reference. File is saved to dir_loc
	'''

	os.chdir(dir_loc)

	with open(new_filename, "w") as f:
		for i, name in enumerate(decoy_names):
			f.write(f"{name},{decoy_sasas[i]},{total_monomer_area}\n")

	f.close()
	return 0

def main():
	'''
	Run the code
	'''

	## Parse the args

	args = parser.parse_args()
	pdb_name = args.pdb
	file_indicator = args.file_indicator
	exhaustive_bool = args.exhaustive
	run_num = args.i
	
	if exhaustive_bool:
		
		## Directories
		
		sample_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/exhaustive_{pdb_name}/{pdb_name}_{run_num}_exhaustive/"
		
		save_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/exhaustive_{pdb_name}/exhaustive_scores_{pdb_name}"
		if not isdir(save_dir):
			os.system(f"mkdir {save_dir}")
			
		monomer_dir = f"/home/jas485/pdb_targets"
		
		## Set up the FreeSASA params as needed
	
		custom_params = freesasa.Parameters({"n-slices" : 100}) ## Makes all of the SASA values much more consistent (within 2-3 Angstroms squared for each complex)
	
		## Include Hydrogens
		freesasa.Structure.defaultOptions["hydrogen"] = True
	
		## Get the monomer total surface area first
	
		monomers = sorted([f for f in listdir(monomer_dir) if isfile(join(monomer_dir, f)) and "_H.pdb" in f and pdb_name in f and "complex" not in f])
		os.chdir(monomer_dir)
		total_monomer_area = 0
		for mono in monomers:
			structure = freesasa.Structure(mono)
			result = freesasa.calc(structure, custom_params)
			total_monomer_area += result.totalArea()
			
		## Calculate SASA for all decoys
		
		filename_mod = "exhaustive"
		
		final_filename = f"interface_areas_{filename_mod}_{pdb_name}_{run_num}.txt"
		
		sampled_names, sampled_sasas = calculate_sasa_many(sample_dir, file_indicator, freesasa_params = custom_params)
	
		write_output_file(final_filename, sampled_names, sampled_sasas, total_monomer_area, dir_loc = save_dir)

		
	elif not exhaustive_bool:

		## Establish the directory
	
		home_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}"
		sampled_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}"
		random_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/sampled_{pdb_name}/random_negatives"
		monomer_dir = f"/home/jas485/pdb_targets"
		save_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/scores_sampled_{pdb_name}"
	
		## Set up the FreeSASA params as needed
	
		custom_params = freesasa.Parameters({"n-slices" : 100}) ## Makes all of the SASA values much more consistent (within 2-3 Angstroms squared for each complex)
	
		## Include Hydrogens
		freesasa.Structure.defaultOptions["hydrogen"] = True
	
		## Get the monomer total surface area first
	
		monomers = sorted([f for f in listdir(monomer_dir) if isfile(join(monomer_dir, f)) and "_H.pdb" in f and pdb_name in f and "complex" not in f])
		os.chdir(monomer_dir)
		total_monomer_area = 0
		for mono in monomers:
			structure = freesasa.Structure(mono)
			result = freesasa.calc(structure, custom_params)
			total_monomer_area += result.totalArea()
	
		## Calculate SASA for all sampled decoys
	
		sampled_names, sampled_sasas = calculate_sasa_many(sampled_dir, file_indicator, freesasa_params = custom_params)
	
		sampled_filename = f"interface_areas_sampled_{pdb_name}.txt"
		write_output_file(sampled_filename, sampled_names, sampled_sasas, total_monomer_area, dir_loc = save_dir)
	
		## Calculate SASA for all random decoys
	
		random_names, random_sasas = calculate_sasa_many(random_dir, file_indicator, freesasa_params = custom_params)
	
		random_filename = f"interface_areas_random_{pdb_name}.txt"
		write_output_file(random_filename, random_names, random_sasas, total_monomer_area, dir_loc = save_dir)

	print(f"Finished with interface area calculation for {pdb_name}")

if __name__ == '__main__':
	main()





