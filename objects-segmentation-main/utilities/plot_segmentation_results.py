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



    


input_folder = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_images']
)

label_folder = os.path.join(
    config['Dataset']['paths']['path_dataset'],
    config['Dataset']['paths']['path_segmentations_npy']
)

results_folder = os.path.join(
    '/petrobr/algo360/current/lvc/objects-segmentation/',
    config['General']['path_predicted_images'],
    # 'segmentations'
    # 'segmentation_results_old_model_new_inference'
    # 'fixednew_protocol/segmentation_results_rgb_masked',
    # 'fixednew_protocol/segmentation_results_rgb',
    # 'fixednew_protocol_inferenceresize/segmentations_rgb_masked',
    # 'finetuning2/segmentation_results_rgb'
    # 'ignore_index/segmentation_results_rgb'
    'finetune_ignore_index/segmentation_results_rgb'

)

plot_folder = os.path.join(
    '/petrobr/algo360/current/lvc/objects-segmentation/',
    config['General']['path_predicted_images'],
    'segmentation_plots'
)


colors = []
class_names = []
for class_id in range(len(config['Dataset']['classes'])):
    color = config['Dataset']['classes'][str(class_id)]['color_rgb']
    color = [x / 255. for x in color]
    colors.append(tuple(color))
    class_names.append(config['Dataset']['classes'][str(class_id)]['name'])


for index, row in tqdm.tqdm(df.iterrows()):
    name = os.path.join(row['Category'], row['File_Names'])
    path = os.path.join(results_folder, name)
    # print("path: ", path)
    predicted = cv2.imread(path.replace('.jpg', '.png'))
    predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)

    input_path = os.path.join(input_folder, name)
    # print("input_path: ", input_path)
    input_im = cv2.imread(input_path)
    input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB)

    label_path = os.path.join(label_folder, name)
    label = cv2.imread('.'.join(label_path.split('.')[:-1]) + '.jpg')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)


    # print("predicted.shape: ", predicted.shape)
    # print("input_im.shape: ", input_im.shape)
    # plot 3 subplots. 1st is the original image, 2nd is the ground truth, 3rd is the predicted image
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(input_im)
    ax[1].imshow(label)
    ax[2].imshow(predicted)

    ax[0].set_title("Input Image")
    ax[1].set_title("Ground Truth")
    ax[2].set_title("Predicted Image")

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].plot([0, 10],[0, 10], alpha=0.)

    for idx, (color, class_name) in enumerate(zip(colors, class_names)):
        ax[3].add_patch(Rectangle((1, (len(colors) - idx) - len(colors)/2), 2, 0.7,
                edgecolor = 'black',
                facecolor = color,
                fill=True,
                lw=1))
        ax[3].text(3.5, (len(colors) - idx) - len(colors)/2 + 0.1, class_name, bbox=dict(facecolor='red', alpha=0.))
    ax[3].axis('off')

    save_path = os.path.join(plot_folder, name)
    # print("save_path: ", save_path)
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()
