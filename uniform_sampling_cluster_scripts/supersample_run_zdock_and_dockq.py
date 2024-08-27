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
import time

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--run", help="Number of the zdock run")
parser.add_argument("--pdb", help="The 4 character pdb ID with only lowercase letters")

## Coordinates function

def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                     [ 0, m.cos(theta),-m.sin(theta)],
                     [ 0, m.sin(theta), m.cos(theta)]])
 
def Ry(theta):
    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                     [ 0           , 1, 0           ],
                     [-m.sin(theta), 0, m.cos(theta)]])
 
def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                     [ m.sin(theta), m.cos(theta) , 0 ],
                     [ 0           , 0            , 1 ]]) 

def get_random_rotation_mat():
    '''
    Returns a random rotation matrix
    to use for modifying all of the coordinates in the PDB.
    Rotation is some random rotation in all 3
    principal directions between 0 and 2 pi
    '''
    phi = np.pi * 2 * random.random()
    theta = np.pi * 2 * random.random()
    psi = np.pi * 2 * random.random()

    return Rz(psi) * Ry(theta) * Rx(phi)
   
def save_pdb(structure, pdb_filename):
    '''
    Saves a PDB file that has been transformed
    '''
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_filename)
    return

def get_rotated_pdb(pdb_file, pdb_name_new, translation, pdb_path = "./", mv_dir = "./"):
    '''
    Input the pdb filename and the path (optional).
    The output is an Nx3 matrix of the coordinates for
    each heavy atom in the pdb_file.
    '''
    os.chdir(pdb_path)
    parser = PDBParser()
   
    ## Get PDB File
    pdb_id = pdb_file[3:7]
    structure  = parser.get_structure(pdb_id, pdb_file)
   
    ## Initialize random rotation
    random_rot = get_random_rotation_mat()
   
    ## Transform all the atomic coordinates
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if "H" not in atom.get_coord():
                        ## Transform the coordinate only with random rotation
                        atom.transform(random_rot, translation)
                        atom.set_coord(np.array(atom.get_coord())[0])
                       
    save_pdb(structure, pdb_name_new)
    
    ## Move the file to the intended directory
    os.system(f"mv {pdb_name_new} {mv_dir}")
    return 0


def get_rotated_target_monomers(pdb_name, test_num, mv_dir):
    '''
    Runs the rotation code and makes a new set of decoys
    '''

    ## Path to all targets on cluster
    target_path = "/home/jas485/pdb_targets"

    ## Sets up the random translation
    trans_arr = np.linspace(15,30, 101)
    rand_x = np.random.choice(trans_arr)
    rand_y = np.random.choice(trans_arr)
    rand_z = np.random.choice(trans_arr)

    ## Gets the randomly rotated and translated files

    subunits = sorted([f for f in listdir(target_path) if isfile(join(target_path, f)) if pdb_name in f and "complex" not in f and "noH" in f])
    chains = []
    new_subunits = []
    for subunit in subunits:
        chain = subunit.split("_")[1]
        chains.append(chain)
        new_pdb_name = f"{pdb_name}_{chain}_noH_rot_{test_num}.pdb"
        new_subunits.append(new_pdb_name)
        translation = np.array((rand_x,rand_y,rand_z), "f")
        get_rotated_pdb(subunit, new_pdb_name, translation, pdb_path = target_path, mv_dir = mv_dir)


    ## Return subunits and chains so it will be easier to call ZDOCK

    subunits = [join(target_path, subunits[0]), join(target_path, subunits[1])]
    return subunits, new_subunits, chains


def run_zdock_on_target(n, chains, receptor, ligand, rot_dir, pdb_name, run_num, save_dir):
    '''
    Runs ZDOCK on the target of interest. Returns as many 
    decoys as are specified in n. Rec and ligand are both
    file names for the receptor and ligand. Chain is
    determined by the output of get_rotated_target_monomers.
    Rot_dir is where the receptor and ligand files are located.
    '''

    ## Create directory to save all of the data

    run_zdock_dir = join(save_dir, f"{pdb_name}_{run_num}_zdock")
    if not isdir(run_zdock_dir): ## Create the zdock directory if it doesn't already exist
        os.system(f"mkdir {run_zdock_dir}")

    os.chdir(run_zdock_dir) ## Change directory into the zdock directory
    os.chdir("../") ## go back one directory because this code blows

    chain_one = chains[0]
    chain_two = chains[1]

    ## Simply define the run_id so there are no random errors
    run_id = f"{pdb_name}_{run_num}"

    receptor_pdb = join(rot_dir, receptor)
    receptor_m_pdb = run_id + "_zdock/receptor_m.pdb"
    ligand_pdb = join(rot_dir, ligand)
    ligand_m_pdb = run_id + "_zdock/ligand_m.pdb"

    os.system("./mark_sur " + receptor_pdb + " " + receptor_m_pdb)
    os.system("./mark_sur " + ligand_pdb + " " + ligand_m_pdb)
    os.system("cp create.pl " + run_id + "_zdock/create.pl")
    os.system("cp create_lig " + run_id + "_zdock/create_lig")
    os.system("cp zdock " + run_id + "_zdock/zdock")

    os.chdir(run_id + "_zdock")

    os.system("./zdock -R receptor_m.pdb -L ligand_m.pdb -o zdock.out -D -N " + str(n))

    os.system("perl create.pl zdock.out")

    print("DONE RUNNING ZDOCK")

    ## Return the directory where all the files are saved
    return run_zdock_dir


def make_dockq_taskfile(pdb_name, taskfilename = "temp.sh", taskfile_save_dir = "./", decoy_dir = "./", dockq_dir = "./", save_dir = "./", native_path = "./", outfile_partial_name = "dockq_output.txt"):
    '''
    Creates a taskfile that can be used to run the DockQ job
    '''
    
    os.chdir(taskfile_save_dir)
    taskfile = open(taskfilename, "w")
    
    all_files = sorted([f for f in listdir(decoy_dir) if isfile(join(decoy_dir, f)) and ".pdb" in f and "ligand" not in f and "receptor" not in f])

    FILES_PER_RUN = 2000
    num_runs = len(all_files)//FILES_PER_RUN
    last_run = len(all_files)%FILES_PER_RUN

    ## If the last run has some files in it, then make a run for them
    if last_run > 0:
        num_runs += 1

    pdb_id = pdb_name

    for i in range(num_runs):
        outfile_name = f"{pdb_id}_zdock_{i}_{outfile_partial_name}"
        if i == num_runs-1:
            ## The last run
            ## Use the number in the last run as the basis!
            if last_run == 0:
                last_run = FILES_PER_RUN
            ## The last run
            ## Use the number in the last run as the basis!
            command = f"python run_dockq_script_enum.py --decoy_dir {decoy_dir}\
            --dockq_dir {dockq_dir} --outfilename {outfile_name} --native_path {native_path} --save_dir {save_dir}\
            --start {i*FILES_PER_RUN} --end {i*FILES_PER_RUN+last_run}\n"
            taskfile.write(command)
            print(command)
            break
        command = f"python run_dockq_script_enum.py --decoy_dir {decoy_dir}\
        --dockq_dir {dockq_dir} --outfilename {outfile_name} --native_path {native_path} --save_dir {save_dir}\
        --start {i*FILES_PER_RUN} --end {i*FILES_PER_RUN+FILES_PER_RUN}\n"
        taskfile.write(command)

    taskfile.close()

def create_dimer_from_monomers(receptor, ligand, pdb_name, save_dir = "./"):
    '''
    Creates a dimer with no hydrogens from the receptor and ligand
    '''

    os.chdir(save_dir)

    with open(f"{pdb_name}_complex.pdb", "w") as complex_pdb:
        receptor_file = open(receptor, "r")
        ligand_file = open(ligand, "r")
        for line in receptor_file:
            if "END" in line: ## DON'T INCLUDE THIS OTHERWISE DOCKQ DOESN'T WORK YOU IDIOT
                continue
            complex_pdb.write(line)
        for line in ligand_file:
            complex_pdb.write(line)


    complex_pdb.close()
    ligand_file.close()
    receptor_file.close()
    ## Return the name of the complex file for later use
    return f"{pdb_name}_complex.pdb"

def create_job_script(job_file_name, tasklist_name, pdb_name, run_num):
    '''
    Creates the job script that will be used to run the DockQ scores
    for the 54,000 decoys
    '''

    with open(job_file_name, 'w') as f:
        f.write(
        f'#!/bin/bash\
        \n#SBATCH --partition=scavenge\
        \n#SBATCH --job-name=nested_{pdb_name}_{run_num}\
        \n#SBATCH -N 1\
        \n#SBATCH -n 1\
        \n#SBATCH -c 1\
        \n#SBATCH --array=1-27\
        \n#SBATCH -Q\
        \n#SBATCH --mem-per-cpu=9999\
        \n#SBATCH -t 3:00:00\
        \n#SBATCH --mail-type=ALL\
        \n#SBATCH --mail-user=jake.sumner@yale.edu\
        \n#SBATCH --output=submit_nest.out\
        \n#SBATCH --requeue\
        \n# run the command\
    \n\
        \n## Loading in the conda environment\
        \nmodule load miniconda\
        \nconda activate general_env\
    \n\
    \n\
        \ncd $SLURM_SUBMIT_DIR\
        \nsed -n "${{SLURM_ARRAY_TASK_ID}}p" {tasklist_name} | /bin/bash')
    f.close()

    return 0

def get_datetime():
    dt = datetime.now()
    str_dt = dt.strftime("%H:%M:%S on %d %B, %Y")
    return str_dt


def main():

    ## Get arguments parsed

    args = parser.parse_args()

    pdb_name = args.pdb

    time.sleep(int(args.run))

    ## Base Dir

    base_dir = f"/gpfs/gibbs/pi/ohern/jas485/supersampling/{pdb_name}/"
    if not isdir(base_dir): ## Create the new directory if it doesn't already exist
        os.system(f"mkdir {base_dir}")

    ## Create the randomly rotated PDB Structures
    
    new_pdb_dir = f"{base_dir}{pdb_name}_{args.run}"
    if not isdir(new_pdb_dir): ## Create the new directory if it doesn't already exist
        os.system(f"mkdir {new_pdb_dir}")
    
    old_subunits, new_subunits, chains = get_rotated_target_monomers(pdb_name, args.run, new_pdb_dir)
    receptor, ligand = new_subunits[:2]

    ## Create a complex from the two subunits that can be used as a reference for DockQ

    old_receptor, old_ligand = old_subunits

    complex_filename = create_dimer_from_monomers(old_receptor, old_ligand, pdb_name, save_dir = new_pdb_dir)

    str_dt = get_datetime()
    print(f"DONE WITH CREATING ROTATE MONOMERS at {str_dt}")

    ## Copy ZDOCK files from source into the new_pdb_dir

    os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/zdock_source/* {new_pdb_dir}")

    ## Run ZDOCK on the randomly rotated chains

    DECOYS_TO_RETURN = 54000
    run_zdock_dir = run_zdock_on_target(DECOYS_TO_RETURN, chains, receptor, ligand, new_pdb_dir, pdb_name, args.run, new_pdb_dir)

    str_dt = get_datetime()
    print(f"DONE WITH ZDOCK at {str_dt}")

    ## Make a taskfile that will run DockQ in parallel

    dockq_save_dir = join(run_zdock_dir, "dockq_output")
    if not isdir(dockq_save_dir):
        os.system(f"mkdir {dockq_save_dir}")

    dockq_tasklist_name = f"zdock_dockq_{args.run}.sh"
    make_dockq_taskfile(
        pdb_name,
        taskfilename = dockq_tasklist_name, 
        taskfile_save_dir = new_pdb_dir, 
        decoy_dir = run_zdock_dir, 
        dockq_dir = "/gpfs/gibbs/pi/ohern/jas485/DockQ/", 
        save_dir = dockq_save_dir, 
        native_path = join(new_pdb_dir, complex_filename), 
        outfile_partial_name = "dockq_output.txt")

    ## Make the job file to run the DockQ scoring

    os.chdir(new_pdb_dir)
    dockq_job_name = f"{pdb_name}_{args.run}_dockq_job.sh"
    create_job_script(dockq_job_name, dockq_tasklist_name, pdb_name, args.run)

    ## Copy over the dockq python script

    os.system(f"cp /gpfs/gibbs/pi/ohern/jas485/DockQ_job_run/run_dockq_script_enum.py {new_pdb_dir}")

    ## Run DockQ on the files to score them

    os.system(f"sbatch {dockq_job_name}")

    str_dt = get_datetime()
    print(f"DONE WITH SCRIPT at {str_dt}")

## Run the script
if __name__ == '__main__':
    main()
    
