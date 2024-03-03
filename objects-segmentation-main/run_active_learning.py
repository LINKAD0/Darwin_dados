import json
import argparse
import os
from src.ActiveLearning import ActiveLearner
from src.utils import boolean_string
import time
import pdb
if __name__ == "__main__":
    app_desc = 'Active learning script'
    parser = argparse.ArgumentParser(
        description = app_desc, 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True)

    # parser.add_argument('-output_path', type=str, default = '/petrobr/algo360/current/corrosion-detector-main/output/')
    # parser.add_argument('-image_path', type=str, default = '/petrobr/algo360/current/corrosion-detector-main/output/2D_images/')
    parser.add_argument('-output_path', type=str, default = 'output/')
    parser.add_argument('-image_path', type=str, default = 'output/2D_images/')

    parser.add_argument('-random_percentage', type=float, default=0)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-beta', type=int, default=2)

    parser.add_argument('-diversity_method', 
        type=str, default='cluster', help='None, cluster, distance_to_train')
    
    parser.add_argument('-copy_2D_images', type=boolean_string, default=False)

    parser.add_argument('-cubemap_keyword', type=str, default="cubemap")
    parser.add_argument('-exp_id', type=int, default=0)
    parser.add_argument('-spatial_buffer', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    args = vars(args)

    t0 = time.time()
    activeLearner = ActiveLearner(args)
    activeLearner.loadData()
    activeLearner.run()

    print("Time: ", time.time() - t0)