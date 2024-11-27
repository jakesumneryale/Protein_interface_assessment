import os
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pandas as pd
import argparse
from sklearn.svm import SVR
import pickle

## Initialize Parser

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", help="PDB that will be left out from the run itself")
parser.add_argument("--kernel", default = "rbf", help="The kernel that will be used for the SVR model")

def perform_svr_loo(corr_dict, pdb_to_exclude, cols, kernel):
    '''
    Performs support vector regression
    using Scikit-learn in a leave-one-out
    fashion, where the pdb_to_exclude is left
    out of the trained model. The 'cols' are used
    for the training.  
    '''
    
    ## Store data in dataframe to numpy array
    data_x = [[],[]]
    data_y = []     
    count = [0,0]
    
    test_x = []
    
    ## Z normalize the data
    for temp_pdb in list(corr_dict.keys()):
        if temp_pdb not in pdb_to_exclude:
            corr_df = corr_dict[temp_pdb]
            for i in range(len(cols)):
                temp_data = np.array(corr_df[cols[i]])
                z_normalized = (temp_data - np.mean(temp_data))/np.std(temp_data)
                z_normalized = list(z_normalized)
                count[i] += len(z_normalized)
                data_x[i] += z_normalized ## Add to the proper list of data variables
                
            ## Get temp_y (dockq) data and normalize
                
            temp_data_y = np.array(corr_df["DockQ"])
            z_normalized = (temp_data_y - np.mean(temp_data_y))/np.std(temp_data_y)
            z_normalized = list(z_normalized)
            data_y += z_normalized
        
        elif temp_pdb in pdb_to_exclude:
            ## The PDB is the one that we are testing on
            corr_df = corr_dict[temp_pdb]
            for i in range(len(cols)):
                temp_data = np.array(corr_df[cols[i]])
                z_normalized = (temp_data - np.mean(temp_data))/np.std(temp_data)
                z_normalized = list(z_normalized)
                test_x.append(z_normalized)
    
    ## Perform SVR
    data_x = np.array(data_x).T
    data_y = np.array(data_y)
    svr = SVR(kernel = kernel)
    svr.fit(data_x, data_y) ## Fit to DockQ
    test_x = np.array(test_x).T
    print(test_x.shape)
    return svr.predict(test_x) ## Test on the leave one out and return the results

def main():
    '''
    RUN THE CODE
    '''
    args = parser.parse_args()
    pdb_to_exclude = args.pdb
    kernel_fxn = args.kernel

    save_dir = "/gpfs/gibbs/pi/ohern/jas485/supersampling/physical_score_svr/svr_2_score_results_no_overlap"

    ## Load in the score dict

    score_loc = "/gpfs/gibbs/pi/ohern/jas485/supersampling/physical_score_svr/uniform_84_score_dataset_overlaps_removed_svr.pickle"

    with open(score_loc, "rb") as f:
        ss_score_dict = pickle.load(f)

    ## Run the LOO code

    svr_cols = ["Intertwined", "Contact_45"]
    output_points = perform_svr_loo(ss_score_dict, pdb_to_exclude, svr_cols, kernel_fxn)

    ## Save the output from the LOO code into a file

    os.chdir(save_dir)

    decoy_names = list(ss_score_dict[pdb_to_exclude]["Decoy"])
    
    new_score_dict = {
        "decoy" : decoy_names,
        "svr_score" : output_points
    }

    new_filename = f"svr_loo_{kernel_fxn}_{pdb_to_exclude}_results.csv"
    
    new_score_df = pd.DataFrame(new_score_dict)
    
    new_score_df.to_csv(new_filename)


if __name__ == '__main__':
    main()
