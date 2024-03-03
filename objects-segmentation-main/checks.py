import os
from src.utils import save_to_csv
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from src.dataset import HilaiDataset
import pdb
def check_processed_images(path_input, path_output,
    path_csv, filename_csv = 'non_processed_images_inference.csv'):

    list_input_files = [x.split('/')[-1] for x in path_input]
    list_output_files = os.listdir(path_output)
    print(list_input_files)
    print(list_output_files)
    pdb.set_trace()

    non_processed_images = []
    for input_image in list_input_files:
        if input_image in list_output_files:
            continue
        else:
            non_processed_images.append(input_image)
    
    if non_processed_images:
        print("Found non processed images. Saving in {}".format(
            os.path.join(path_csv, filename_csv)
        ))
        save_to_csv(non_processed_images, path_csv, 
            filename_csv)
    return non_processed_images

def check_processed_images_from_dataset(cfg):
    print("Checking processed images...")
    dataset_val = HilaiDataset(cfg)

    check_processed_images(dataset_val.paths_images, 
        os.path.join(cfg['path_output'], cfg['path_segmentations']),
        cfg['path_output'])


parser = ArgumentParser()
# ==== Path parameters
parser.add_argument('-filename', type=str, default="output/cub_maps_split")
parser.add_argument('-path_output', type=str, default="output")


parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-filename_ext', type=str, default=".png")
parser.add_argument('-path_segmentations', type=str, default='corrosion')
parser.add_argument('-split_train', type=float, default=0.)
parser.add_argument('-split_val', type=float, default=0.)
parser.add_argument('-split_test', type=float, default=1.)
parser.add_argument('-split', type=str, default='test')
parser.add_argument('-use_reference', type=bool, default=False)
parser.add_argument('-ignore_already_processed', type=bool, default=False)

args = parser.parse_args()

args = vars(args)


check_processed_images_from_dataset(args)
