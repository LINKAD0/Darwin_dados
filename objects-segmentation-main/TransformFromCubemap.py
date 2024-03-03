from argparse import ArgumentParser

import src.cubemap as cm
import os, imageio
import pdb
import time
import pandas as pd
from pathlib import Path

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="output/corrosion/")
parser.add_argument('-path_output_2D', type=str, default="output/2D_predictions/")
parser.add_argument('-path_csv', type=str, default="output/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="output/corrosion_360/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])
parser.add_argument('-n_jobs', type=int, default=1)

parser.add_argument('-load_split_dataset', type=bool, default=False)
parser.add_argument('-split_root', type=str, default="output/splits")
parser.add_argument('-data_split', type=int, default=0)
'''
parser.add_argument('-path_input_cubemap_segmentation', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion/")
parser.add_argument('-path_output_2D', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/2D_predictions/")
parser.add_argument('-path_csv', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/")
parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
parser.add_argument('-path_output_360', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/corrosion_360/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])
parser.add_argument('-n_jobs', type=int, default=1)

parser.add_argument('-load_split_dataset', type=bool, default=True)
parser.add_argument('-split_root', type=str, default="/petrobr/algo360/current/corrosion-detector-main/output/splits")
parser.add_argument('-data_split', type=int, default=None)
'''
# parser.add_argument('-path_output_split', type=str, default="output/cub_maps_split/")


args = parser.parse_args()
# print(vars(args))
args = vars(args)



t0 = time.time()

def init_folders(args):
    if not os.path.exists(args['path_output_2D']):
        os.makedirs(args['path_output_2D'])

    if not os.path.exists(args['path_output_360']):
        os.makedirs(args['path_output_360'])


    with open(os.path.join(
        os.path.dirname(args['path_csv']),"unsuccessful_from_cubemap.txt"), 'w') as f:
        f.write('')

def load_filenames_360_from_csv(config):
    csv_path = os.path.join(config['split_root'],"tmp_data_split_from_cubemap_{}.csv".format(config['data_split']))
    filenames_360 = pd.read_csv(csv_path)['filename'].tolist()
    return filenames_360

if __name__ == "__main__":
    # %%
    print("Starting cubemap to 360 conversion...")
    t_start = time.time()
    init_folders(args)
    # Create the cubmap prediction (each folder contains six images)
    # path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'



    if args['load_split_dataset'] == True:
        filenames_360 = load_filenames_360_from_csv(args)
    else:
        # Read CSV with list of inference 360 images
        filenames_360 = ['_'.join(os.path.basename(i).split('_')[1:]) for i in Path(args['path_input_cubemap_segmentation']).glob('**/*.png')]
        print(filenames_360)
        filenames_360 = cm.get_unique_from_cubemaps(filenames_360)
        print(filenames_360)
        filenames_360 = list(set(filenames_360))
        # Transform cubemap faces to 2D cubemap representation
        print('total of {} input files'.format(len(filenames_360)))

        # ignore already processed files
        filenames_360 = cm.ignore_already_processed_cubemaps(filenames_360, args['path_output_360'])
        print(filenames_360)

    if args['mode'] == 'xprojector':


        print('number of pending files: {}'.format(len(filenames_360)))

        cm.cubemaps_to_360(args['path_input_cubemap_segmentation'], args['cubemap_keyword'], 
            filenames_360, args['path_output_360'], args['path_csv'], n_jobs=args['n_jobs'])
                

            
        print("...Finished cubemap to 360 conversion. Time:", time.time() - t0)
    elif args['mode'] == 'custom':

        print("Starting cubemap to 2D conversion...")

        # %%
        # Create the cubmap prediction (each folder contains six images)
        # path_cub_prediction = root_path + 'activeLearningLoop-main/output/cub_predictions/'

        if not os.path.exists(args['path_output_2D']):
            os.makedirs(args['path_output_2D'])
            
        for i in range(0, len(filenames_360)):
            print('image: ', i)
            args_ = (args['path_input_cubemap_segmentation'], args['cubemap_keyword'], 
                filenames_360[i], args['path_output_2D'])
            cm.cubemap_to_2D(args_)
                
        print("...Finished cubemap to 2D conversion. Time:", t0 - time.time())

        # %%

        print("Starting 2D to 360 conversion...")

        if not os.path.exists(args['path_output_360']):
            os.makedirs(args['path_output_360'])
                
        img_pred = cm.return_files(args['path_output_2D'])
        print(img_pred)

        # Transform each cubmap prediction into a 360 image prediction
        for i in range(0, len(img_pred)):
            print(i)
            print(args['path_output_2D'] + img_pred[i], args['path_output_360'] + img_pred[i])
            cm.convert_img(args['path_output_2D'] + img_pred[i], args['path_output_360'] + img_pred[i])
            

        print("...Finished 2D to 360 conversion. Time:", t0 - time.time())

    t_end = round(time.time() - t_start, 2)
    