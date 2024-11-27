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
parser.add_argument("--kernel", default = "rbf", help="The kernel that will be used for the SVR model")

def perform_svr_loo(corr_dict, test_dict, cols, kernel, save_dir = "."):
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
    
    ## Z normalize the data and create the training set from the original 84 (78 corrected) targets
    for temp_pdb in list(corr_dict.keys()):
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
        
    ## Train the SVR
    data_x = np.array(data_x).T
    data_y = np.array(data_y)
    svr = SVR(kernel = kernel)
    svr.fit(data_x, data_y) ## Fit to DockQ
    
    os.chdir(save_dir)
            
    ## Create the training set from the 62 from ZDOCK benchmark 5.5 and test it
    for temp_pdb in list(test_dict.keys()):
        
        test_df = test_dict[temp_pdb]
        
        test_x = []
        
        for i in range(len(cols)):
            temp_data = np.array(test_df[cols[i]])
            z_normalized = (temp_data - np.mean(temp_data))/np.std(temp_data)
            z_normalized = list(z_normalized)
            test_x.append(z_normalized)
    
        ## Test the SVR on the models
        test_x = np.array(test_x).T
        output_points = svr.predict(test_x)
        
        ## Save the data
        
        decoy_names = list(test_dict[temp_pdb]["Decoy"])
        
        new_score_dict = {
            "decoy" : decoy_names,
            "svr_score" : output_points
        }
        
        new_score_df = pd.DataFrame(new_score_dict)
        
        new_filename = f"bm_55_svr_results_{temp_pdb}.csv"
        
        new_score_df.to_csv(new_filename)
        
    return  

def main():
    '''
    RUN THE CODE
    '''
    args = parser.parse_args()
    kernel_fxn = args.kernel

    save_dir = "/gpfs/gibbs/pi/ohern/jas485/supersampling/physical_score_svr/svr_2_score_results_bm55_test"

    ## Load in the score dict

    score_loc = "/gpfs/gibbs/pi/ohern/jas485/supersampling/physical_score_svr/uniform_84_score_dataset_overlaps_removed_svr.pickle"
    bm55_loc = "/gpfs/gibbs/pi/ohern/jas485/supersampling/physical_score_svr/zdock_benchmark_55_score_dict_with_physical.pickle"

    with open(score_loc, "rb") as f:
        ss_score_dict = pickle.load(f)
        
    with open(bm55_loc, "rb") as f:
        bm55_score_dict = pickle.load(f)

    ## Run the SVR code

    svr_cols = ["Intertwined", "Contact_45"]
    
    perform_svr_loo(ss_score_dict, bm55_score_dict, svr_cols, kernel_fxn)


if __name__ == '__main__':
    main()
