import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.dataset import HilaiDataset, ObjectSegmentationDataset
from glob import glob
from argparse import ArgumentParser
import pdb
from pytorch_lightning.callbacks import Callback
import numpy as np
import sys
# sys.path.append('segmentation_models_ptorch')
# import segmentation_models_pytorch_custom as smpc
import segmentation_models_pytorch as smp

from src.uncertainty import get_uncertainty_map
import os
from src.utils import create_dir, create_output_folders, save_to_csv
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import time
import torch.nn.functional as F
import pandas as pd, json

t_start = time.time()

def getPadding(len):
    if len % 16 != 0:
        padding = 16 - len % 16
    else:
        padding = 0
    return padding
def getImagePadding(X):
    padding = [getPadding(X.shape[-2]),
                getPadding(X.shape[-1])]
    
    X = F.pad(X, (0, padding[1], 0, padding[0]), mode='constant', value=0)
    return X

colors = {
            "0": {
                "name": "Background",
                "color": [0,0,0],
                "color_rgb": [0,0,0]
            },
            "1": {
                "name": "Equipamento",
                "color": [1,1,1],
                "color_rgb": [123, 213, 207]
            },
            "2": {
                "name": "Escadas",
                "color": [2,2,2],
                "color_rgb": [229, 114, 238]
            },
            "3": {
                "name": "Estrutura",
                "color": [3,3,3],
                "color_rgb": [198, 21, 21]
            },
            "4": {
                "name": "Flanges",
                "color": [4,4,4],
                "color_rgb": [218, 71, 209]
            },
            "5": {
                "name": "Guarda Corpo",
                "color": [5,5,5],
                "color_rgb": [232, 234, 87]
            },
            "6": {
                "name": "Piso",
                "color": [6,6,6],
                "color_rgb": [19, 27, 129]
            },
            "7": {
                "name": "Suportes",
                "color": [7,7,7],
                "color_rgb": [233, 117, 2]
            },
            "8": {
                "name": "Teto",
                "color": [8,8,8],
                "color_rgb": [208, 77, 9]
            },
            "9": {
                "name": "Tubulacao",
                "color": [9,9,9],
                "color_rgb": [68, 231, 134]
            },
            "10": {
                "name": "Sem Categoria",
                "color": [10,10,10],
                "color_rgb": [255, 255, 255]
            }

        }
# define the LightningModule
class LitModel(pl.LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # fill arguments
        
        parser.add_argument('-path_model', type=str, default="models")
        parser.add_argument('-exp_id', type=int, default=0)

        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg


        self.model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet',
                in_channels=3, classes=cfg['class_n'])

        path_model = os.path.join(cfg['path_model'], self.model.__class__.__name__ + 
            '_' + str(cfg['exp_id']) + '.p')
        print(path_model)

        self.model.load_state_dict(
            torch.load(path_model, map_location=torch.device(cfg['accelerator']))['model_state_dict']
        )

        self.transform_image = transforms.Compose([   
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])                  

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        x, _, filenames = batch

        if self.cfg['get_uncertainty'] == True:
            encoder_features, aspp_features, y = self.model(x)
            encoder_features = encoder_features.mean((2, 3))
            aspp_features = aspp_features.mean((2, 3))
        else:
            y = self.model(x)
            encoder_features = None
            aspp_features = None
        y = torch.nn.functional.softmax(y, dim=1)
        y = y.cpu().detach().numpy()
        segmentations = np.argmax(y, axis=1).astype(np.uint8)
        if self.cfg['get_uncertainty'] == True:
            y = y[:, 1]
            # Use metrics module to calculate uncertainty metric
            uncertainty_map = get_uncertainty_map(np.expand_dims(y, axis=-1))
            
            uncertainty = np.mean(uncertainty_map, axis=(1, 2))   
            # Logging to TensorBoard by default
            # self.log("train_loss", loss)
        else:
            uncertainty_map = None
            uncertainty = None
        return {'softmax': y, 'segmentations': segmentations, 
            'uncertainty_map': uncertainty_map, 'uncertainty': uncertainty,
            'encoder_features': encoder_features, 
            'aspp_features': aspp_features, 'filenames': filenames}
        # return y

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        if self.cfg['get_uncertainty'] == True:        
            x, _, filenames = batch
        else:
            x, filenames = batch


        x = getImagePadding(x)
        if self.cfg['get_uncertainty'] == True:
            encoder_features, aspp_features, y = self.model(x)
            encoder_features = encoder_features.mean((2, 3))
            aspp_features = aspp_features.mean((2, 3))
        else:
            y = self.model(x)
            encoder_features = None
            aspp_features = None
        y = torch.nn.functional.softmax(y, dim=1)
        y = y.cpu().detach().numpy()
        segmentations = np.argmax(y, axis=1).astype(np.uint8)
        # print("segmentations.shape", segmentations.shape)
        if self.cfg['get_uncertainty'] == True:
            y = y[:, 1]
            # Use metrics module to calculate uncertainty metric
            uncertainty_map = get_uncertainty_map(np.expand_dims(y, axis=-1))
            
            uncertainty = np.mean(uncertainty_map, axis=(1, 2))   
            # Logging to TensorBoard by default
            # self.log("train_loss", loss)
        else:
            uncertainty_map = None
            uncertainty = None        
        return {'softmax': y, 'segmentations': segmentations, 
            'uncertainty_map': uncertainty_map, 'uncertainty': uncertainty,
            'encoder_features': encoder_features, 
            'aspp_features': aspp_features, 'filenames': filenames}
        # return y


import multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler
class HilaiDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HilaiDataModule")
        # fill arguments

        parser.add_argument('-filename', type=str, default="/petrobr/algo360/current/lvc/objects-segmentation-main/output/cub_maps_split/")

        parser.add_argument('-filename_ext', type=str, default=".png")

        parser.add_argument('-path_output', type=str, default="output")

        parser.add_argument('-split_train', type=float, default=0.)
        parser.add_argument('-split_val', type=float, default=0.)
        parser.add_argument('-split_test', type=float, default=1.)

        parser.add_argument('-split', type=str, default='test')
        parser.add_argument('-use_reference', type=bool, default=False)

        parser.add_argument('-path_segmentations', type=str, default='segmentations_objects')
        parser.add_argument('-path_uncertainty', type=str, default='uncertainty')
        parser.add_argument('-path_mean_uncertainty', type=str, default='mean_uncertainty')
        parser.add_argument('-path_uncertainty_map', type=str, default='uncertainty_map')
        parser.add_argument('-path_encoder_features', type=str, default='encoder_features')
        parser.add_argument('-path_aspp_features', type=str, default='aspp_features')

        parser.add_argument('-test_csv_name', type=str, default='inference_csv')
        parser.add_argument('-mean_uncertainty_csv_name', type=str, default='mean_uncertainty')

        parser.add_argument('-test_batch_size', type=int, default=6)

        parser.add_argument('-split_root', type=str, default="output/splits")
        parser.add_argument('-data_split', type=int, default=0)



        return parent_parser
        
    def __init__(self, **cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.num_replicas = int(cfg['gpus'])*int(cfg['num_nodes'])
        
        # self.data_dir = data_dir
        # self.batch_size = batch_size
    def setup(self, stage: str):
        self.dataset_val = HilaiDataset(self.cfg)
        

    def val_dataloader(self):
        #sampler = DistributedSampler(self.dataset_val,num_replicas=self.num_replicas,shuffle=False,)
        #return DataLoader(self.dataset_val,sampler=None, batch_size=self.cfg['test_batch_size'],num_workers=mp.cpu_count()//len(os.listdir(self.cfg['split_root'])))
        return DataLoader(self.dataset_val,sampler=None, batch_size=self.cfg['test_batch_size'],num_workers=0)

    def test_dataloader(self):
        #sampler = DistributedSampler(self.dataset_val,num_replicas=self.num_replicas,shuffle=False,)
        #return DataLoader(self.dataset_val,sampler=None, batch_size=self.cfg['test_batch_size'],num_workers=mp.cpu_count()//len(os.listdir(self.cfg['split_root'])))
        return DataLoader(self.dataset_val,sampler=None, batch_size=self.cfg['test_batch_size'],num_workers=0)



parser = ArgumentParser()
# add PROGRAM level args

parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-gpus', type=int, default=1)
parser.add_argument('-accelerator', type=str, default='cuda')
parser.add_argument('-strategy', type=str, default='ddp')
parser.add_argument('-max_epochs', type=int, default=300)
parser.add_argument('-num_nodes', type=int, default=1)
parser.add_argument('-sync_batchnorm', type=bool, default=True)

parser.add_argument('-get_uncertainty', type=bool, default=False)
parser.add_argument('-class_n', type=int, default=10)

parser.add_argument('-use_360_images', type=bool, default=False)



# add model specific args
parser = LitModel.add_model_specific_args(parser)
parser = HilaiDataModule.add_model_specific_args(parser)
# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
#parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# pdb.set_trace()
#print(vars(args))

class SaveOutcomesCallback(Callback):
    def __init__(self,**args):
        self.args = args
    def on_validation_start(self, trainer, pl_module):
        self.validation_filenames = []
        self.uncertainty_mean_values = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        '''
        print(outputs['softmax'].shape)
        print(outputs['segmentations'].shape)
        print(outputs['uncertainty_map'].shape)
        print(outputs['uncertainty'].shape)
        print(outputs['encoder_features'].shape)
        print(outputs['filenames'])
        '''

        args = self.args

        x, filenames = batch

        for idx in range(x.shape[0]):
            filename = filenames[idx].split('/')[-1].split('\\')[-1] # .replace('\\', '/') # aqui poner carpeta de clase
            self.validation_filenames.append(filename)
            self.uncertainty_mean_values.append(outputs['uncertainty'][idx])

            np.savez(args['path_output'] +'/'+ args['path_encoder_features'] +'/'+ filename.split('.')[0] + '.npz', 
                outputs['encoder_features'].cpu().detach().numpy()[idx])
            np.savez(args['path_output'] +'/'+ args['path_aspp_features'] +'/'+ filename.split('.')[0] + '.npz', 
                outputs['aspp_features'].cpu().detach().numpy()[idx])
            np.savez(args['path_output'] +'/'+ args['path_uncertainty_map'] +'/'+ filename.split('.')[0] + '.npz', 
                outputs['uncertainty_map'][idx])
            
            cv2.imwrite(os.path.join(args['path_output'], args['path_segmentations'], filename), outputs['segmentations'][idx])

            
    def on_validation_end(self, trainer, pl_module):

        # Save CSV with 360 image names
        self.validation_filenames_360 = [x.split('.')[0] for x in self.validation_filenames]
        self.validation_filenames_360 = ['_'.join([x.split('_')[1],x.split('_')[2]]) for x in self.validation_filenames_360]
        self.validation_filenames_360 = list(dict.fromkeys(self.validation_filenames_360))
        #print(self.validation_filenames_360)

        save_to_csv(self.validation_filenames_360, 
            args['path_output'],
            args['test_csv_name'] + '.csv')
        

        # Save CSV with mean uncertainty

        save_to_csv(zip(self.validation_filenames, self.uncertainty_mean_values), 
            args['path_output'],
            args['mean_uncertainty_csv_name'] + '.csv')

    def on_test_start(self, trainer, pl_module):
        if args['get_uncertainty'] == True:
            self.validation_filenames = []
        if args['get_uncertainty'] == True:
            self.uncertainty_mean_values = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        '''
        print(outputs['softmax'].shape)
        print(outputs['segmentations'].shape)
        print(outputs['uncertainty_map'].shape)
        print(outputs['uncertainty'].shape)
        print(outputs['encoder_features'].shape)
        print(outputs['filenames'])
        '''

        args = self.args

        x, filenames = batch

        for idx in range(x.shape[0]):
            filename = filenames[idx].split('/')[-1].split('\\')[-1]
            if args['get_uncertainty'] == True:
                self.validation_filenames.append(filename)

                self.uncertainty_mean_values.append(outputs['uncertainty'][idx])

                np.savez(args['path_output'] +'/'+ args['path_encoder_features'] +'/'+ filename.split('.')[0] + '.npz', 
                    outputs['encoder_features'].cpu().detach().numpy()[idx])
                np.savez(args['path_output'] +'/'+ args['path_aspp_features'] +'/'+ filename.split('.')[0] + '.npz', 
                    outputs['aspp_features'].cpu().detach().numpy()[idx])
                np.savez(args['path_output'] +'/'+ args['path_uncertainty_map'] +'/'+ filename.split('.')[0] + '.npz', 
                    outputs['uncertainty_map'][idx])

                with open(args['path_output'] +'/'+ args['path_mean_uncertainty'] +'/'+ filename.split('.')[0] + '.csv', 
                    'w', newline='') as order_csv:
                    order_csv_write = csv.writer(order_csv)
                    order_csv_write.writerow([os.path.basename(filename).split('.')[0], outputs['uncertainty'][idx]])

            
            save_segmentation_colors = True
            output = outputs['segmentations'][idx]
            if save_segmentation_colors == True:
                output = outputs['segmentations'][idx] + 1

                output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
                for class_id in range(len(colors)):
                    color = colors[str(class_id)]['color_rgb']
                    mask = output[:, :, 0] == class_id
                    output[mask] = color
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args['path_output'], args['path_segmentations'], filename), 
                        output)
                
    def on_test_end(self, trainer, pl_module):
        if self.args['use_360_images'] == True:

            # Save CSV with 360 image names
            self.validation_filenames_360 = [x.split('.')[0] for x in self.validation_filenames]
            self.validation_filenames_360 = ['_'.join([x.split('_')[1],x.split('_')[2]]) for x in self.validation_filenames_360]
            self.validation_filenames_360 = list(dict.fromkeys(self.validation_filenames_360))
            #print(self.validation_filenames_360)

            save_to_csv(self.validation_filenames_360, 
                self.args['path_output'],
                self.args['test_csv_name'] + '.csv')
        
        if args['get_uncertainty'] == True:

            # Save CSV with mean uncertainty

            save_to_csv(zip(self.validation_filenames, self.uncertainty_mean_values), 
                self.args['path_output'],
                self.args['mean_uncertainty_csv_name'] + '.csv')

#trainer = pl.Trainer.from_argparse_args(args, callbacks=[SaveOutcomesCallback()],
#    gpus=-1)



args = vars(args)

import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpus'])

#print(args['gpus'],args['accelerator'],args['strategy'])
trainer = pl.Trainer(
    devices=1,
    accelerator=args['accelerator'],
    strategy=None,#args['strategy'],
    #max_epochs=args['max_epochs'],
    #logger=logger,
    callbacks=[SaveOutcomesCallback(**args)],
    #sync_batchnorm=args['sync_batchnorm'],
    #num_sanity_val_steps=-1,
    #accumulate_grad_batches=args['accumulate_grad_batches'],
    #gradient_clip_val=0.5,
    #gradient_clip_algorithm="value",
    #replace_sampler_ddp=False
    )



create_output_folders(args)

# init the model
model = LitModel(**args)


dm = HilaiDataModule(**args)
#dm.setup("validation")
#val_dataloader = dm.val_dataloader()
#trainer.validate(model, dm)
trainer.test(model, dm)
t_end = round(time.time() - t_start, 2)
print("Inference time", t_end)

