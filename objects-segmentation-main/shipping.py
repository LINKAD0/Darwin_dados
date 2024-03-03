from argparse import ArgumentParser

import os

import glob
import shutil

parser = ArgumentParser()
# add PROGRAM level args

parser.add_argument('-path_corrosion_360', type=str, default="output/corrosion_360/",
    help="Corrosion 360 image folder")
parser.add_argument('-path_corrosion_360_platform', type=str, default="output/corrosion_360_platform/",
    help="Output corrosion 360 image folder with each platform in a different folder")

args = parser.parse_args()
args = vars(args)

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print('dir exists')
    pass

def to_platform_folder(path, output_path):

    files = glob.glob(os.path.join(path, '*.png'))
    # print(files)
    files = [os.path.basename(x).split('_')[0] for x in files]
    unique_platforms = list(set(files))
    print("Platform number: ", len(unique_platforms))
    # print(unique_platforms)
    for platform in unique_platforms:
        platform_path = os.path.join(output_path, platform)
        mkdir(platform_path)
        files_from_platform = glob.glob(os.path.join(path, '{}_*.png'.format(platform)))
        for file in files_from_platform:
            platform_filename = os.path.basename(file).split('_')[-1]
            shutil.copyfile(file, os.path.join(platform_path, platform_filename))


mkdir(args['path_corrosion_360_platform'])
to_platform_folder(args['path_corrosion_360'], args['path_corrosion_360_platform'])
