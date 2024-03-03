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
print("label_path: ", label_path)

out_folder = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    "labels_png"
)
pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
for category in df['Category'].unique():
    pathlib.Path(os.path.join(out_folder, category)).mkdir(parents=True, exist_ok=True)



def save_npy_as_png(args):
    row, label_path, out_folder = args
    path = os.path.join(label_path, row['Category'], '.'.join(row['File_Names'].split('.')[:-1]) + '.npy')

    npy_label = np.expand_dims(np.load(path).astype(np.uint8), axis=-1)

    npy_label = cv2.cvtColor(npy_label, cv2.COLOR_GRAY2RGB)
    im = Image.fromarray(npy_label)

    im.save(os.path.join(out_folder,
        row['Category'],
        "{}.png".format('.'.join(row['File_Names'].split('.')[:-1]))
        )
    )

args = []

for index, row in tqdm.tqdm(df.iterrows()):
    args.append((row, label_path, out_folder))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(save_npy_as_png, args), total=len(args)):
    pass