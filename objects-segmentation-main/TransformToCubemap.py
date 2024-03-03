from argparse import ArgumentParser

import src.cubemap as cm
import pdb
import time
import os

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('-path_360_images', type=str, default="C:/Users/jchamorro/Downloads/P67/P67/sample_360_images/")
parser.add_argument('-path_output', type=str, default="output/cub_maps/")
parser.add_argument('-path_cubemap_images', type=str, default="output/cub_maps_split/")
parser.add_argument('-mode', type=str, default="xprojector", choices=['xprojector', 'custom'])


args = parser.parse_args()
# print(vars(args))
args = vars(args)
if __name__ == "__main__":
    t_start = time.time()
    if not os.path.exists(args['path_cubemap_images']):
        os.makedirs(args['path_cubemap_images'])
    if not os.path.exists(args['path_360_images']):
        os.makedirs(args['path_360_images'])


    if args['mode'] == 'xprojector':
        # 360 images to cubmaps, path_360_images contains all the RGB images
        cm.x_generate_cubmaps(args['path_360_images'], args['path_cubemap_images'], dims=(1344, 1344))
    elif args['mode'] == 'custom':

        # 360images to cubmaps, path_360_images contains all the RGB images
        cm.generate_cubmaps(args['path_360_images'], args['path_output'])

        # Split cubemaps into 6 images
        cm.split_cub_imgs(args['path_output'], args['path_cubemap_images'])

    t_end = round(time.time() - t_start, 2)
    print("Execution time:", t_end)