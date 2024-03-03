from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd
from pathlib import Path

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemaps', type=str, default="output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_images/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-split_root', type=str, default="output/splits")

'''
parser.add_argument('-path_input_cubemaps', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/cub_maps_split/")
parser.add_argument('-path_output_2D', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/2D_predictions/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-split_root', type=str, default="output/splits")
'''

parser.add_argument('-load_split_dataset', type=bool, default=True)
parser.add_argument('-data_split', type=int, default=0)

args = parser.parse_args()

args = vars(args)


def load_filenames_360_from_csv(config):
    csv_path = os.path.join(config['split_root'],"tmp_data_split_cubemap_to_2d_{}.csv".format(config['data_split']))
    filenames_360 = pd.read_csv(csv_path)['filename'].tolist()
    return filenames_360

if __name__ == "__main__":
    # %%
    time_start = time.time()

    if args['load_split_dataset'] == True:
        filenames_360 = load_filenames_360_from_csv(args)
    else:
        
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


    cm.cubemaps_to_2D(args['path_input_cubemaps'], args['cubemap_keyword'], 
            filenames_360, args['path_output_2D'])
    
    time_end = time.time() - time_start
    print("Execution time:",time_end)