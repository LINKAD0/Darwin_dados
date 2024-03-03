import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
# from icecream import ic
from PIL import Image
import pdb
from scipy.special import softmax
from tqdm.contrib import tzip

from src.utils import create_dir, getImagePadding
from src.dataset import show
import src.uncertainty as uncertainty

import sys, pdb
sys.path.append('segmentation_models_ptorch')

import segmentation_models_pytorch as smp
# import segmentation_models_pytorch_dropout as smpd
# import segmentation_models_pytorch_custom as smpc

import time

from sklearn import metrics
import src.ActiveLearning as al
import copy
import src.utils as utils
from src.dataset import Corrosion2DDataset

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import jaccard_score
def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model
def check_dropout_enabled(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(m.training)
    return model

class Predictor(object):
    def __init__(self, config, test_dataset):
        self.config = config
        self.type = self.config['General']['type']
        self.save_images = self.config['Inference']['save_images']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        # resize = 513

        if config['General']['model_type'] == 'unet':        
            
            self.model = smp.Unet('xception', encoder_weights='imagenet', in_channels=3,
                encoder_depth=4, decoder_channels=[128, 64, 32, 16], classes=self.config['Dataset']['num_classes'])
            # path_model = os.path.join(config['General']['path_model'], 'Unet.p')


        elif config['General']['model_type'] == 'deeplab':        
            network = smp.DeepLabV3Plus   
            
            self.model = network('tu-xception41', encoder_weights='imagenet', in_channels=3,
                    classes=self.config['Dataset']['num_classes'])                
            # path_model = os.path.join(config['General']['path_model'], 'DeepLabV3Plus.p')

        elif config['General']['model_type'] == 'deeplab_dropout':        
            
            self.model = smp.DeepLabV3Plus('resnet34', encoder_weights='imagenet', in_channels=3,
                classes=self.config['Dataset']['num_classes'])
            # path_model = os.path.join(config['General']['path_model'], 'DeepLabV3Plus.p')

        
        path_model = os.path.join('/petrobr/algo360/current/lvc/objects-segmentation/',
                                  self.config['General']['path_model'], self.model.__class__.__name__ + 
            '_' + str(self.config['General']['exp_id']) + '.p')
        print("path_model: %s" % path_model)


        # path_model = os.path.join(config['General']['path_model'], 'ResUnetPlusPlus.p')
        
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()

        '''
        if config['General']['model_type'] == 'deeplab':   
            dropout_ = DropoutHook(prob=0.2)
            # self.model.apply(dropout_.register_hook)

            print(self.model.encoder.model.blocks_1.stack)

            # self.model.encoder.model.blocks_1.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_4.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_7.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_10.apply(dropout_.register_hook)
            self.model.encoder.model.blocks_12.apply(dropout_.register_hook)
        '''

        
        self.output_dir = self.config['General']['path_predicted_images']
        create_dir(self.output_dir)
        # create_dir(os.path.join(self.output_dir, "segmentation_results"))

        self.path_dir_segmentation = os.path.join(self.output_dir, 'segmentation_results')
        self.path_dir_segmentation_rgb = os.path.join(self.output_dir, 'segmentation_results_rgb')
        self.path_dir_segmentation_rgb_masked = os.path.join(self.output_dir, 'segmentation_results_rgb_masked')

        create_dir(self.path_dir_segmentation)
        
        print("self.path_dir_segmentation", self.path_dir_segmentation)
        
        for key, value in self.config['Dataset']['classes'].items():
            create_dir(os.path.join(self.path_dir_segmentation, value['name']))
        for key, value in self.config['Dataset']['classes'].items():
            create_dir(os.path.join(self.path_dir_segmentation_rgb, value['name']))
        for key, value in self.config['Dataset']['classes'].items():
            create_dir(os.path.join(self.path_dir_segmentation_rgb_masked, value['name']))

        self.test_dataset = test_dataset

    def run(self):
        with torch.no_grad():
            # list_data = self.config['Dataset']['paths']['list_datasets']

            print(len(self.test_dataset.paths_images))
            
            # idxs_reduced = np.random.choice(np.arange(len(self.test_dataset.paths_images)), size=12, replace=False)
            # self.test_dataset.paths_images = [self.test_dataset.paths_images[idx] for idx in idxs_reduced]

            test_dataloader = DataLoader(self.test_dataset, batch_size=self.config['General']['test_batch_size'], shuffle=False)

        
            self.inferDataLoader(test_dataloader)

    def getUncertaintyBatch(self, softmax_segmentation):
        ## print(softmax_segmentation.shape)
        pred_entropy_batch = []
        if self.config['ActiveLearning']['spatial_buffer'] == True:
            buffer_mask_batch = []
        for idx in range(len(softmax_segmentation)):
            pred_entropy = uncertainty.get_uncertainty_map2(
                    np.expand_dims(softmax_segmentation[idx], axis=-1)).astype(np.float32)
            
            
            if self.config['ActiveLearning']['spatial_buffer'] == True:
                pred_entropy, buffer_mask = uncertainty.apply_spatial_buffer(
                    pred_entropy, softmax_segmentation[idx]
                )
                buffer_mask_batch.append(np.expand_dims(buffer_mask, axis=0))
            
            pred_entropy_batch.append(np.expand_dims(pred_entropy, axis=0))
        pred_entropy_batch = np.concatenate(pred_entropy_batch, axis=0)
        # print("pred_entropy_batch.shape", pred_entropy_batch.shape)
        return pred_entropy_batch, buffer_mask_batch


    def inferDataLoader(self, dataloader, getEncoder = False):
        pbar = tqdm(dataloader)
        pbar.set_description("Testing")
        self.model.to(self.device)

        softmax_segmentations = []
        output_values = []
        uncertainty_values = []
        reference_values = []
        encoder_values = []

        if self.config['ActiveLearning']['spatial_buffer'] == True:
            self.buffer_mask_values = []
        for i, (X, Y_segmentations, path) in enumerate(pbar):
            # X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)            
            X = X.to(self.device)          

            # print("Y_segmentations.shape", Y_segmentations.shape)
            # print("summary", torch.cuda.memory_summary(device=None, abbreviated=False))

            X, original_shape = getImagePadding(X)

            if getEncoder == True:
                # print(len(self.model(X)))
                # pdb.set_trace()
                encoder_features, output_segmentations = self.model(X)
                encoder_features = encoder_features.mean((2, 3))
                del X
            else:
                output_segmentations = self.model(X)
                del X

            # unpad
            output_segmentations = output_segmentations[:, :, :original_shape[-2], :original_shape[-1]]
            assert output_segmentations.shape[-2:] == Y_segmentations.shape[-2:]
            
            softmax_segmentation = output_segmentations.cpu().detach().numpy()

            output = softmax_segmentation.argmax(axis=1).astype(np.uint8)
            
            Y_segmentations = Y_segmentations.squeeze(1).detach().numpy()
                
            '''
            print("path", path)
            print("output.shape", output.shape)
            print(np.unique(output, return_counts=True))
            print(self.output_dir)

            '''
            if self.config['General']['ignore_background'] == True:
                output = output + 1
            # print("output grayscale shape", output.shape)
            save_path = os.path.join(self.path_dir_segmentation, 
                                     os.path.dirname(path[0]).split('/')[-1], os.path.basename(path[0]))
            # print("save_path", save_path)
            # save output image with cv2
            output = np.squeeze(output)
            cv2.imwrite(save_path, output)
            

            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
            for class_id in range(len(self.config['Dataset']['classes'])):
                color = self.config['Dataset']['classes'][str(class_id)]['color_rgb']
                mask = output[:, :, 0] == class_id
                output[mask] = color

            # print("output rgb shape", output.shape)
            # print("np.unique(predicted, return_counts=True): ", np.unique(output, return_counts=True))
            save_path = os.path.join(self.path_dir_segmentation_rgb,
                                     os.path.dirname(path[0]).split('/')[-1], os.path.basename(path[0]))
            # print("output.shape", output.shape)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, output)
            
            Y_segmentations = np.squeeze(Y_segmentations)

            for chan in range(3):
                output[:, :, chan][Y_segmentations == 0] = 0

            save_path = os.path.join(self.path_dir_segmentation_rgb_masked, 
                            os.path.dirname(path[0]).split('/')[-1], os.path.basename(path[0]))
            cv2.imwrite(save_path, output)

class PredictorWithMetrics(Predictor):
    def __init__(self, config, test_dataset):
        super().__init__(config, test_dataset.paths_images)
        self.test_dataset = test_dataset
        # check_dropout_enabled(self.model)
        # exit(0)


    def run(self):
        
        # list_data = self.config['Dataset']['paths']['list_datasets']

        print(len(self.test_dataset.paths_images))
        
        # idxs_reduced = np.random.choice(np.arange(len(self.test_dataset.paths_images)), size=12, replace=False)
        # self.test_dataset.paths_images = [self.test_dataset.paths_images[idx] for idx in idxs_reduced]

        test_dataloader = DataLoader(self.test_dataset, batch_size=self.config['General']['test_batch_size'], shuffle=False)

    
        output_values, reference_values, uncertainty_values = self.inferDataLoader(
            test_dataloader)

        if self.config['General']['ignore_background'] == True:
            output_values = output_values + 1
            reference_values = reference_values + 1
            reference_values[reference_values == self.config['Dataset']['num_classes'] +1] = 0
        # print(np.unique(output_values, return_counts=True))
        # print(np.unique(reference_values, return_counts=True))
        print("output_values.shape", output_values.shape)
        print("len(self.test_dataset.paths_images)", len(self.test_dataset.paths_images))

        print("np.unique(output_values), np.unique(reference_values)", 
              np.unique(output_values, return_counts=True), 
              np.unique(reference_values, return_counts=True))
        if self.save_images == True:   

            path_dir_segmentation = os.path.join(self.output_dir, 'segmentations')
            path_dir_segmentation_rgb = os.path.join(self.output_dir, 'segmentations_rgb')
            path_dir_segmentation_rgb_masked = os.path.join(self.output_dir, 'segmentations_rgb_masked')

            create_dir(path_dir_segmentation)
            
            print("path_dir_segmentation", path_dir_segmentation)
            
            colormap = []
            for key, value in self.config['Dataset']['classes'].items():
                colormap.append(value['color_rgb'])
                create_dir(os.path.join(path_dir_segmentation, value['name']))
                create_dir(os.path.join(path_dir_segmentation_rgb, value['name']))
                create_dir(os.path.join(path_dir_segmentation_rgb_masked, value['name']))



            for value, reference_value, path in tzip(output_values, reference_values, self.test_dataset.paths_images):
                class_folder_name = path.split('/')[-2]


                original_im = Image.open(path)
                original_size = original_im.size


                
                value_int = cv2.cvtColor(value, cv2.COLOR_GRAY2RGB)
                # print(np.unique(value_int, return_counts=True))
                value_int = cv2.resize(value_int, original_size, interpolation=cv2.INTER_NEAREST)
                # print(np.unique(value_int, return_counts=True))
                if np.any(value_int == 0):
                    print("np.unique(value_int, return_counts=True)", np.unique(value_int, return_counts=True))
                    pdb.set_trace()
                save_path = os.path.join(path_dir_segmentation, class_folder_name, os.path.basename(path).replace('.jpg', '.png'))
                # print("path", path)
                # print(np.unique(value_int, return_counts=True))
                cv2.imwrite(save_path, value_int)
                # pdb.set_trace()
                # value_int.save(os.path.join(path_dir_segmentation, class_folder_name, os.path.basename(path)))


                value_rgb = cv2.cvtColor(value, cv2.COLOR_GRAY2RGB)
                for idx, color in enumerate(colormap):
                    for chan in range(3):
                        value_rgb[:, :, chan][value == idx] = color[chan]
                

                value_rgb = transforms.ToPILImage()(value_rgb).resize(original_size, resample=Image.NEAREST)
                value_rgb.save(os.path.join(path_dir_segmentation_rgb, class_folder_name, os.path.basename(path).replace('.jpg', '.png')))

                path_reference = path.replace('images', 'labels_png').replace('.jpg', '.png')
                reference = np.squeeze(cv2.imread(path_reference, 0))


                value_rgb = np.array(value_rgb)

                for chan in range(3):
                    value_rgb[:, :, chan][reference == 0] = 0

                value_rgb = transforms.ToPILImage()(value_rgb).resize(original_size, resample=Image.NEAREST)
                value_rgb.save(os.path.join(path_dir_segmentation_rgb_masked, class_folder_name, os.path.basename(path).replace('.jpg', '.png')))

            del original_im

        output_values = output_values.flatten()
        reference_values = reference_values.flatten()

        output_values = output_values[reference_values != 0]
        reference_values = reference_values[reference_values != 0]

        f1 = metrics.f1_score(reference_values, output_values, average='macro')
        print("f1:", round(f1*100, 1))
        f1_per_class = metrics.f1_score(reference_values, output_values, average=None)
        print("f1:", [round(x*100, 1) for x in f1_per_class])
        
        oa = metrics.accuracy_score(reference_values, output_values)
        print("oa:", round(oa*100, 1))

    def inferDataLoader(self, dataloader, getEncoder = False):
        pbar = tqdm(dataloader)
        pbar.set_description("Testing")
        self.model.to(self.device)

        softmax_segmentations = []
        output_values = []
        uncertainty_values = []
        reference_values = []
        encoder_values = []

        if self.config['ActiveLearning']['spatial_buffer'] == True:
            self.buffer_mask_values = []
        for i, (X, Y_segmentations, path) in enumerate(pbar):
            # X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)            
            X = X.to(self.device)          

            ## print("X.shape", X.shape)
            # print("Y_segmentations.shape", Y_segmentations.shape)
            # print("summary", torch.cuda.memory_summary(device=None, abbreviated=False))

            # X, _ = getImagePadding(X)
            # Y_segmentations, _ = getImagePadding(Y_segmentations)

            if getEncoder == True:
                # print(len(self.model(X)))
                # pdb.set_trace()
                encoder_features, output_segmentations = self.model(X)
                encoder_features = encoder_features.mean((2, 3))
                del X
            else:
                output_segmentations = self.model(X)
                del X

            
            softmax_segmentation = output_segmentations.cpu().detach().numpy()

            output = softmax_segmentation.argmax(axis=1).astype(np.uint8)
            
            Y_segmentations = Y_segmentations.squeeze(1).detach().numpy()

            output_values.append(output)
            reference_values.append(Y_segmentations)

            # ========= Apply softmax
            softmax_segmentation = softmax(softmax_segmentation, axis=1)[:, 1]

            # ========= Get uncertainty   
            if self.config['get_uncertainty'] == True: 
                ## print(softmax_segmentation.shape)
                uncertainty_batch, buffer_mask_batch = self.getUncertaintyBatch(softmax_segmentation)
                # print("pred_entropy_batch.shape", pred_entropy_batch.shape)
                print("uncertainty_batch.get_device()", uncertainty_batch.get_device())

                uncertainty_values.append(uncertainty_batch)


            if self.config['ActiveLearning']['spatial_buffer'] == True:
                buffer_mask_batch = np.concatenate(buffer_mask_batch, axis=0)
                self.buffer_mask_values.append(buffer_mask_batch)

            if getEncoder == True:
                encoder_value = encoder_features.cpu().detach().numpy()
                print("encoder_value.get_device()", encoder_value.get_device())

                encoder_values.append(encoder_value)
        output_values = np.concatenate(output_values, axis=0)
        reference_values = np.concatenate(reference_values, axis=0)
        if self.config['get_uncertainty'] == True:
            uncertainty_values = np.concatenate(uncertainty_values, axis=0)
            print("uncertainty_values.shape", uncertainty_values.shape)
        else:
            uncertainty_values = None
        if self.config['ActiveLearning']['spatial_buffer'] == True:
            self.buffer_mask_values = np.concatenate(self.buffer_mask_values, axis=0)
            print("self.buffer_mask_values.shape", self.buffer_mask_values.shape)

        print(output_values.shape, reference_values.shape) 
        if getEncoder == True:
            encoder_values = np.concatenate(encoder_values, axis=0)
            print("encoder_values.shape", encoder_values.shape)
            return output_values, reference_values, uncertainty_values, encoder_values

        return output_values, reference_values, uncertainty_values  
