import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# import wandb
import cv2
import torch.nn as nn

from tqdm import tqdm
from os import replace
from numpy.core.numeric import Inf

from src.utils import get_losses, get_optimizer, get_schedulers, create_dir

import sys

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn.functional as F
import pdb

from src.utils import getImagePadding

class EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.restartCounter()
    def restartCounter(self):
        self.counter = 0
    def increaseCounter(self):
        self.counter += 1
    def checkStopping(self):
        if self.counter >= self.patience:
            return True
        else:
            return False
class Trainer(object):
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']

        if config['General']['fine_tune'] == False:
            classes = config['Dataset']['num_classes']
        else:
            classes = 2 # corrosion num_classes
        if config['General']['model_type'] == 'unet':        
            
            self.model = smp.Unet('xception', encoder_weights='imagenet', in_channels=3,
                encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=classes)
        elif config['General']['model_type'] == 'deeplab': # use this one        
            
            self.model = smp.DeepLabV3Plus('tu-xception41', encoder_weights='imagenet', in_channels=3,
                classes=classes)


        if config['General']['fine_tune'] == True:
            print("Finetuning from corrosion model...")        
            # path_model = os.path.join('/petrobr/algo360/current/lvc/objects-segmentation/'
            #     self.config['General']['path_model'], 'pretrained', self.model.__class__.__name__ + 
            #     '_' + str(self.config['General']['exp_id']) + '_corrosion.p')
            path_model = config['General']['fine_tune_path_model']
            print("finetuning model path: %s" % path_model)

            self.model.load_state_dict(
                torch.load(path_model, map_location=self.device)['model_state_dict']
            )

            # print(self.model)
            # print(self.model.segmentation_head)
            # pdb.set_trace()

            upsampling = 4
            activation = None
            classes = self.config['Dataset']['num_classes']
            
            self.model.segmentation_head = SegmentationHead(
            in_channels=self.model.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
            )

            '''

            do the trainer with cpu (not to use slurm)
            print self.model and self.model.segmentation_head.blocks_1.stack or something like that

            self.model.fc = nn.Linear(2048, 10) 

            '''
        self.model.to(self.device)

        # print(self.model)
        
        # print(self.model.encoder.model.blocks_1.stack)

        # pdb.set_trace()
        # exit(0)
        # print("input shape: ", (3,resize,resize))
        # print(resize)
        # summary(self.model, (3,resize,resize))
        # exit(0)

        self.loss_depth, self.loss_segmentation = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

        self.path_model = os.path.join(self.config['General']['path_model'], 
            self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']))

            
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']

        val_loss = Inf
        es = EarlyStopping(10)
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch+1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_segmentations, _) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                # print("X shape 1: ", X.shape)
                # X, _ = getImagePadding(X)
                # Y_segmentations, _ = getImagePadding(Y_segmentations)
                # print("X shape 2: ", X.shape)


                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer

                output_segmentations = self.model(X)
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)

                pbar.set_postfix({'training_loss': running_loss/(i+1)})

            new_val_loss = self.run_eval(val_dataloader)

            if new_val_loss < val_loss:
                self.save_model()
                val_loss = new_val_loss
                es.restartCounter()
            else:
                es.increaseCounter()

            self.schedulers[0].step(new_val_loss)

            if es.checkStopping() == True:
                print("Early stopping")
                print(es.counter, es.patience)
                print('Finished Training')
                exit(0)

        print('Finished Training')

    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_segmentations, _) in enumerate(pbar):
                X, Y_segmentations = X.to(self.device), Y_segmentations.to(self.device)


                output_depths, output_segmentations = (None, self.model(X))

                output_depths = output_depths.squeeze(1) if output_depths != None else None
                Y_segmentations = Y_segmentations.squeeze(1)
                if i==0:
                    X_1 = X
                    Y_segmentations_1 = Y_segmentations
                    output_depths_1 = output_depths
                    output_segmentations_1 = output_segmentations
                # get loss
                loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})

        return val_loss/(i+1)

    def save_model(self):
        
        create_dir(self.path_model)
        torch.save({'model_state_dict': self.model.state_dict(),
                    # 'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                    }, self.path_model+'.p')
        print('Model saved at : {}'.format(self.path_model))
