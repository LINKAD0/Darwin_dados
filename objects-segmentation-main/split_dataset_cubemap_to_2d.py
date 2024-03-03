from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd
from pathlib import Path
import numpy as np

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemaps', type=str, default="output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_images/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-data_splits_root', type=str, default="output/splits")

'''
parser.add_argument('-path_input_cubemaps', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/2D_predictions/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-data_splits_root', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/splits")
'''

parser.add_argument('-n_splits', type=int, default=1)

args = parser.parse_args()

args = vars(args)



if __name__ == "__main__":
    # %%
    print("Starting cubemap to 2D conversion...")

    # Create output folder
    if not os.path.exists(args['path_output_2D']):
        os.makedirs(args['path_output_2D'])

    # Get cubemap input files
    filenames = [os.path.basename(str(i)) for i in Path(args['path_input_cubemaps']).glob('*.png')]

    filenames_360 = cm.get_unique_from_cubemaps2(filenames)
    filenames_360 = list(set(filenames_360))

    print('total of {} input files'.format(len(filenames_360)))

    # ignore already processed files
    filenames_360 = cm.ignore_already_processed_cubemaps(filenames_360, args['path_output_2D'])


    print('number of pending files: {}'.format(len(filenames_360)))


    ixs = np.arange(len(filenames_360))
    splits = np.array_split(ixs,args['n_splits'])
    #split_ix = np.random.choice(ixs,n)

    df = pd.DataFrame({'filename':filenames_360})
    splits = np.array_split(df,args['n_splits'])

    for i in range(args['n_splits']):
        df0 = splits[i]
        df0.to_csv(os.path.join(args['data_splits_root'],"tmp_data_split_cubemap_to_2d_{}.csv".format(i)))
