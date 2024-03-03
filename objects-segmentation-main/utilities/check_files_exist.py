import pandas as pd
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pdb
import cv2
import tqdm
import pathlib
import multiprocessing as mp

with open('../config.json', 'r') as f:
    config = json.load(f)


csv_path = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_csv']
)
print("csv_path: ", csv_path)
df = pd.read_csv(csv_path)

label_path = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_segmentations_npy']
)

input_path = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_images']
)

print("label_path: ", label_path)

out_folder = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    "labels_png"
)


def check_file_exists(args):
    row, label_path, input_path = args
    path = os.path.join(label_path, row['Category'], '.'.join(row['File_Names'].split('.')[:-1]) + '.npy')
    input_path = os.path.join(input_path, row['Category'], '.'.join(row['File_Names'].split('.')[:-1]) + '.jpg')

    npy_label = np.expand_dims(np.load(path).astype(np.uint8), axis=-1)
    input_im = cv2.imread(input_path)
    assert os.path.exists(path)
    assert os.path.exists(input_path)
    assert os.path.isfile(path)
    assert os.path.isfile(input_path)


args = []

for index, row in tqdm.tqdm(df.iterrows()):
    args.append((row, label_path, input_path))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(check_file_exists, args), total=len(args)):
    pass