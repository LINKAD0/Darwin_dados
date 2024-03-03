import json
from glob import glob
from src.Predictor import Predictor, PredictorWithMetrics
from src.dataset import Corrosion2DDataset, ObjectSegmentationDataset
import pandas as pd
import os

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import pdb
base_path = '/petrobr/algo360/current/lvc/objects-segmentation/'
with open(os.path.join(base_path, 'config.json'), 'r') as f:
    config = json.load(f)


config['split'] = 'test'
config['seed'] = config['General']['seed']

config['get_uncertainty'] = False



df = pd.read_csv(os.path.join(
    config['Dataset']['paths']['path_dataset'],    
    config['Dataset']['paths']['path_csv'])
)


test_data = ObjectSegmentationDataset(df, config)

# print(test_data.paths_images)
# pdb.set_trace()

if config['Inference']['resize_flag'] == False:
    predictor = Predictor(config, test_data)
else:
    predictor = PredictorWithMetrics(config, test_data)

predictor.run()
