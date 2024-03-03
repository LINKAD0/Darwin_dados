import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from src.Trainer import Trainer
from src.dataset import Corrosion2DDataset, ObjectSegmentationDataset

import time
import pdb
import copy

import src.utils as utils

import pandas as pd
import os

base_path = '/petrobr/algo360/current/lvc/objects-segmentation/'
with open(os.path.join(base_path, 'config.json'), 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

dataset_name = config['Dataset']['paths']['path_dataset']

print("config['Dataset']['num_classes']", config['Dataset']['num_classes'])
print(config['Dataset']['splits'])

config['seed'] = config['General']['seed']

df = pd.read_csv(os.path.join(
    config['Dataset']['paths']['path_dataset'],    
    config['Dataset']['paths']['path_csv'])
)
## train set
# =============================================================================
dataset_config = config   
print(dataset_name, dataset_config['Dataset']['splits'])     
dataset_config['split'] = 'train'

train_data = ObjectSegmentationDataset(df, dataset_config)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True, 
    drop_last=True)

## validation set
# =============================================================================
autofocus_datasets_val = []

dataset_config = config       
dataset_config['split'] = 'val'
val_data = ObjectSegmentationDataset(df, dataset_config)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)
# =============================================================================
trainer = Trainer(config)
t0 = time.time()
trainer.train(train_dataloader, val_dataloader)
print("Execution time", time.time() - t0)