import json
import argparse
import os
from src.utils import boolean_string
import time
import pdb
import csv
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


def load_filenames_from_csv(config):
    csv_path = os.path.join(config['split_root'],"tmp_data_split_uncertainty_{}.csv".format(config['data_split']))
    filenames = pd.read_csv(csv_path)['filename'].tolist()
    return filenames

# Import writer class from csv module

def getMeanUncertaintyFromMap(filenames, output_path, mean_uncertainty_folder):



    
    mean_uncertainty_path = Path(os.path.join(output_path, mean_uncertainty_folder))
    mean_uncertainty_path.mkdir(parents=True, exist_ok=True)
    args = []
    for filename in filenames:
        args.append((filename, mean_uncertainty_path))

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm(pool.imap_unordered(saveMeanInCsv, args), total=len(args)):
        pass


def saveMeanInCsv(args):
    filename, mean_uncertainty_path = args
    uncertainty_map = np.load(filename)['arr_0']
    uncertainty_mean = np.mean(uncertainty_map, axis = (0, 1))
    file_path = mean_uncertainty_path / "{}.csv".format(os.path.basename(filename.split('.')[0]))
    
    with open(file_path, 'w', newline='') as order_csv:
        order_csv_write = csv.writer(order_csv)
        # order_csv_write.writerow(
        #     ["filename", "uncertainty_mean"])
        order_csv_write.writerow([os.path.basename(filename).split('.')[0], uncertainty_mean])

if __name__ == "__main__":
    app_desc = 'Active learning script'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    # parser.add_argument('-output_path', type=str, default = '/petrobr/algo360/current/corrosion-detector-main/output/')
    parser.add_argument('-output_path', type=str, default = 'output/')

    parser.add_argument('-mean_uncertainty_foldername', type=str, default="mean_uncertainty")
    parser.add_argument('-split_root', type=str, default="output/splits")
    parser.add_argument('-data_split', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    args = vars(args)

    t0 = time.time()

    filenames_csv = load_filenames_from_csv(args)
    filenames = [x.replace('csv','npz') for x in filenames_csv]
    filenames = [os.path.join(args['output_path'], 'uncertainty_map', x) for x in filenames]
    
    getMeanUncertaintyFromMap(
        filenames, 
        args['output_path'],
        args['mean_uncertainty_foldername']
    )

    print("Time: ", time.time() - t0)