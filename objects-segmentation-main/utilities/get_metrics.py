import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import cv2
import pathlib
from matplotlib.patches import Rectangle
import tqdm
import random
import pdb
from sklearn import metrics


random.seed(10)
np.random.seed(10)

with open('../config.json', 'r') as f:
    config = json.load(f)




csv_path = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_csv']
)
print("csv_path: ", csv_path)
df = pd.read_csv(csv_path)
df = df[df['Set'] == 2] 

# n_smaller = len(df)*0.1
# df = df.sample(n=int(n_smaller))



    

label_folder = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_segmentations']
)

results_folder = os.path.join(
    '/petrobr/algo360/current/lvc/objects-segmentation/',
    config['General']['path_predicted_images'],
    # 'segmentation_results'
    # 'segmentations'
    # 'fixednew_protocol/segmentation_results_fixednew_protocol',
    # 'fixednew_protocol_inferenceresize/segmentations',
    # 'finetuning_from_corrosion/segmentation_results',
    # 'repetition_fixednew_protocol/segmentation_results',
    # 'finetuning2/segmentation_results',
    # 'ignore_index/segmentation_results',
    # 'finetune_ignore_index/segmentation_results',
    'resize_ignore_index/segmentations',

)

predicted_values = []
label_values = []
print("label folder: ", label_folder)
print("results folder: ", results_folder)

for index, row in tqdm.tqdm(df.iterrows()):
    name = os.path.join(row['Category'], row['File_Names'])
    path = os.path.join(results_folder, name)
    path = '.'.join(path.split('.')[:-1]) + '.png'
    
    # print("path: ", path)
    # pdb.set_trace()
    predicted = cv2.imread(path, 0)
    
    
    label_path = os.path.join(label_folder, name)
    label = cv2.imread('.'.join(label_path.split('.')[:-1]) + '.png', 0)
    

    label = label.flatten()
    predicted = predicted.flatten()
    predicted = predicted[label != 0]
    label = label[label != 0] 

    # print("np.unique(predicted, return_counts=True): ", np.unique(predicted, return_counts=True))
    # print("np.unique(label, return_counts=True): ", np.unique(label, return_counts=True))
    # pdb.set_trace()
    
    predicted_values.append(predicted)
    label_values.append(label)
    # except:
    #     print("Error image. path: ", path)
    # print(np.unique(predicted, return_counts=True))
    # print(np.unique(label, return_counts=True))

    # print("predicted.shape: ", predicted.shape)
    # print("label.shape: ", label.shape)

label_values = np.concatenate(label_values).flatten()
predicted_values = np.concatenate(predicted_values).flatten()


print("label_values.shape: ", label_values.shape)
print("predicted_values.shape: ", predicted_values.shape)

print("np.unique(label_values, return_counts=True): ", np.unique(label_values, return_counts=True))
print("np.unique(predicted_values, return_counts=True): ", np.unique(predicted_values, return_counts=True))

print("label_values.dtype: ", label_values.dtype)
print("predicted_values.dtype: ", predicted_values.dtype)

f1 = metrics.f1_score(label_values, predicted_values, average='macro')
print("f1:", round(f1*100, 1))
f1_per_class = metrics.f1_score(label_values, predicted_values, average=None)
print("f1:", [round(x*100, 1) for x in f1_per_class])

oa = metrics.accuracy_score(label_values, predicted_values)
print("oa:", round(oa*100, 1))
