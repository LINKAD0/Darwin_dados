from src.dataset import ignore_already_computed, get_total_paths
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print('dir exists')
    pass

parser = ArgumentParser()
parser.add_argument('-filename', type=str, default="output/cub_maps_split")

parser.add_argument('-filename_ext', type=str, default=".png")

parser.add_argument('-path_output', type=str, default="output")
parser.add_argument('-data_splits_root', type=str, default="output/splits")
parser.add_argument('-path_images', type=str, default="imgs")
parser.add_argument('-path_segmentations', type=str, default='segmentations_objects')
parser.add_argument('-n_splits', type=int, default=1)


config = vars(parser.parse_args())

mkdir(config['data_splits_root'])

path_images =  config['filename']
path_input = get_total_paths(path_images, config['filename_ext'])
path_output = os.path.join(config['path_output'], config['path_segmentations'])
mkdir(path_output)
print("Input path:", path_images)
print("Output path:", path_output)
files = ignore_already_computed(path_input, path_output)

print("Pending files: ", len(files))
ixs = np.arange(len(files))
splits = np.array_split(ixs,config['n_splits'])
#split_ix = np.random.choice(ixs,n)

df = pd.DataFrame({'filename':files})
splits = np.array_split(df,config['n_splits'])

for i in range(config['n_splits']):
    df0 = splits[i]
    df0.to_csv(os.path.join(config['data_splits_root'],"tmp_data_split_{}.csv".format(i)))
