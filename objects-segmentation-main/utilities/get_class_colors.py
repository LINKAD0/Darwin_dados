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
    config['Dataset']['paths']['path_segmentations']
)
print("label_path: ", label_path)

args = []

for index, row in tqdm.tqdm(df.iterrows()):
    path = os.path.join(label_path, row['Category'], '.'.join(row['File_Names'].split('.')[:-1]) + '.png')
    im = np.array(Image.open(path))
    print("im.shape: ", im.shape)
    print("im unique: ", np.unique(im))
    pdb.set_trace()