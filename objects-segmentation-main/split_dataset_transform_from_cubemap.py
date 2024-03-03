import os
import src.cubemap as cm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print('dir exists')
    pass

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="output/corrosion_360/")
parser.add_argument('-data_splits_root', type=str, default="output/splits")
'''
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion_360/")
'''

parser.add_argument('-n_splits', type=int, default=1)



args = parser.parse_args()
print(vars(args))
args = vars(args)

mkdir(args['data_splits_root'])
mkdir(args['path_output_360'])


# Read CSV with list of inference 360 images
filenames_360 = ['_'.join(os.path.basename(str(i)).split('_')[1:4]) for i in Path(args['path_input_cubemap_segmentation']).glob('**/*.png')]

path_input, cubemap_keyword, path_output_360 = (args['path_input_cubemap_segmentation'], 
            args['cubemap_keyword'], args['path_output_360'])

filenames_360 = cm.get_unique_from_cubemaps(filenames_360)

filenames_360 = list(set(filenames_360))


# Transform cubemap faces to 2D cubemap representation
print('total of {} input files'.format(len(filenames_360)))

# ignore already processed files
filenames_360 = cm.ignore_already_processed_cubemaps(filenames_360, path_output_360)
print('number of pending files: {}'.format(len(filenames_360)))



ixs = np.arange(len(filenames_360))
splits = np.array_split(ixs,args['n_splits'])
#split_ix = np.random.choice(ixs,n)

df = pd.DataFrame({'filename':filenames_360})
splits = np.array_split(df,args['n_splits'])

for i in range(args['n_splits']):
    df0 = splits[i]
    df0.to_csv(os.path.join(args['data_splits_root'],"tmp_data_split_from_cubemap_{}.csv".format(i)))
