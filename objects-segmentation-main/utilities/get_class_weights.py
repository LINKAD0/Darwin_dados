# Get class weights from pixel count
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
from sklearn.utils.class_weight import compute_class_weight

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

total_count = {}

num_classes = config['Dataset']['num_classes']

for i in range(num_classes):
    total_count[i] = 0

print("total_count: ", total_count)

def count_in_sample(args):
    row, label_path, num_classes = args
    count_dict = {}
    for i in range(num_classes):
        count_dict[i] = 0

    path = os.path.join(label_path, row['Category'], '.'.join(row['File_Names'].split('.')[:-1]) + '.png')
    # im = np.load(path)
    im = np.array(Image.open(path))
    uniques, counts = np.unique(im, return_counts=True)
    # print("unique: ", uniques)
    # print("count: ", counts)
    for unique, count in zip(uniques, counts):
        count_dict[unique] = count
    return count_dict


args = []
df = df[df['Set'] == 1] 

for index, row in tqdm.tqdm(df.iterrows()):
    args.append((row, label_path, num_classes))

pool = mp.Pool(mp.cpu_count())

with pool as p:
    r = list(tqdm.tqdm(p.imap(count_in_sample, args), total=len(args)))

# print(len(r))
# print("r[0]: ", r[0])

for count_dict in r:
    for key in count_dict:
        total_count[key] += count_dict[key]
print(total_count)

unique = []
count = []
for key, value in total_count.items():
    unique.append(key)
    count.append(value)

count = count[1:]
max_count = max(count)
print("max_count: ", max_count) 
weights = [round(max_count/c,2) for c in count]
print("weights: ", weights)

percentages = [round(c/np.sum(np.array(count))*100,1) for c in count]
print("percentages: ", percentages)