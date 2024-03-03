import json
import argparse
import os
from src.ActiveLearning import ActiveLearner, getFilenamesFromFolder
from src.utils import boolean_string
import time
import pdb
import numpy as np
import pandas as pd
# from src.dataset import ignore_already_computed

def ignore_already_computed(path_input, path_output):
    
    list_output_files = os.listdir(path_output)
    list_input_files = [os.path.basename(x) for x in path_input]

    reduced_input_files = list(set(list_input_files).difference( set(list_output_files)))

    print('total number of files: {}'.format(len(list_input_files)))
    print('total of images processsed: {}'.format(len(list_output_files)))
    print('total remaining images: {}'.format(len(reduced_input_files)))

    return reduced_input_files


if __name__ == "__main__":
    app_desc = 'Active learning script'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    # parser.add_argument('-output_path', type=str, default = '/petrobr/algo360/current/corrosion-detector-main/output/')
    parser.add_argument('-output_path', type=str, default = 'output/')

    parser.add_argument('-exp_id', type=int, default=0)

    parser.add_argument('-k', type=int, default=3)

    parser.add_argument('-data_splits_root', type=str, default="output/splits")
    parser.add_argument('-n_splits', type=int, default=1)

    
    args = parser.parse_args()
    print(args)
    args = vars(args)

    t0 = time.time()
    activeLearner = ActiveLearner(args)
    filenames = getFilenamesFromFolder(os.path.join(activeLearner.config['output_path'], 'uncertainty_map'))
    filenames = [os.path.basename(x) for x in filenames]

    filenames = [x.replace('.npz', '.csv') for x in filenames]
    
    filenames = ignore_already_computed(filenames, 
        os.path.join(activeLearner.config['output_path'], 'mean_uncertainty'))


    ixs = np.arange(len(filenames))
    splits = np.array_split(ixs,args['n_splits'])
    #split_ix = np.random.choice(ixs,n)

    df = pd.DataFrame({'filename':filenames})
    splits = np.array_split(df,args['n_splits'])

    for i in range(args['n_splits']):
        df0 = splits[i]
        df0.to_csv(os.path.join(args['data_splits_root'],"tmp_data_split_uncertainty_{}.csv".format(i)))
