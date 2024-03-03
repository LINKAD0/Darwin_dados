import os
import random
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import get_total_paths
import pdb
import pandas as pd
import cv2
from src.Custom_augmentation import ToMask
import torchvision

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def get_transforms(self, config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
        ToMask(config['Dataset']['classes']),
    ])
    return transform_image, transform_seg



Dataset.get_transforms = get_transforms

class HilaiDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- config -: json config file
            :- input_folder_path -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config):

        self.config = config

        
        self.split = config['split']
        input_folder_path = config['filename']
        self.use_reference = config['use_reference']
        self.path_output = os.path.join(config['path_output'], config['path_segmentations'])
        path_images = input_folder_path # config['path_images']

        if self.split != 'test':
            
            path_images = os.path.join(config['Dataset']['paths']['path_dataset'], config['dataset_name'], config['Dataset']['paths']['path_images'])
            path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], config['dataset_name'], config['Dataset']['paths']['path_segmentations'])

            self.paths_images = get_total_paths(path_images, config['Dataset']['extensions']['ext_images'])
            self.paths_segmentations = get_total_paths(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])
            print("N. of input images: {}, N. of reference images: ".format(len(self.paths_images), len(self.paths_segmentations)))

            assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
            assert (config['Dataset']['splits']['split_train']+config['Dataset']['splits']['split_test']+config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"                

        """
        self.paths_images = get_total_paths(path_images, config['filename_ext'])
        #print(self.paths_images)

        self.path_images_all = self.paths_images
        
        self.paths_images = ignore_already_computed(self.paths_images, self.path_output)
        """
        csv_path = os.path.join(config['split_root'],"tmp_data_split_{}.csv".format(config['data_split']))
        self.paths_images = pd.read_csv(csv_path)['filename'].tolist()
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        #print(len(self.paths_images))




        # check for segmentation

        # ic(input_folder_path, self.paths_images)
        # utility func for splitting

        self.paths_images = self.get_splitted_dataset(config, self.split, input_folder_path, self.paths_images)

        print('{} :{}'.format(self.split,len(self.paths_images)))

        # Get the transforms
        self.transform_image, self.transform_seg = self.get_transforms(config)

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if self.split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if self.split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if self.split=='train' else 0
        self.resize = 512

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / depth maps and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(Image.open(self.paths_images[idx]))
        # ic(Image.open(self.paths_depths[idx]).mode)
        # ic(Image.open(self.paths_segmentations[idx]).mode)
        # depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        if self.use_reference == True:
            segmentation = self.transform_seg(Image.open(self.paths_segmentations[idx]))
        # imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            # depth = TF.hflip(depth)
            if self.use_reference == True:
                segmentation = TF.hflip(segmentation)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize-1)
            max_size = self.resize - random_size
            left = int(random.random()*max_size)
            top = int(random.random()*max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            # depth = TF.crop(depth, top, left, random_size, random_size)
            if self.use_reference == True:
                segmentation = TF.crop(segmentation, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            # depth = transforms.Resize((self.resize, self.resize))(depth)
            if self.use_reference == True:
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        if random.random() < self.p_rot:
            #rotate
            random_angle = random.random()*20 - 10 #[-10 ; 10]
            mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            # depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            if self.use_reference == True:
                segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
            #crop to remove black borders due to the rotation
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left,top)
            size = self.resize - 2*coin
            image = TF.crop(image, coin, coin, size, size)
            # depth = TF.crop(depth, coin, coin, size, size)
            if self.use_reference == True:
                segmentation = TF.crop(segmentation, coin, coin, size, size)
            #Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            #depth = transforms.Resize((self.resize, self.resize))(depth)
            if self.use_reference == True:
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)
        # show([imgorig, image, depth, segmentation])
        # exit(0)
        if self.use_reference == True:
            return image, segmentation, self.paths_images[idx]
        else:
            return image, self.paths_images[idx]


    def get_splitted_dataset(self, config, split, input_folder_path, path_images, path_segmentation = None):
        list_files = [os.path.basename(im) for im in path_images]
        np.random.seed(config['seed'])
        np.random.shuffle(list_files)
        if split == 'train':
            selected_files = list_files[:int(len(list_files)*config['split_train'])]# [:100]
            # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
        elif split == 'val':
            selected_files = list_files[int(len(list_files)*config['split_train']):int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val'])]
            # selected_files = getFilesWithoutBlankReference(dataset_name, selected_files)
        else:
            selected_files = list_files[int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val']):]# [:100]

        print('Train list len', len(list_files[:int(len(list_files)*config['split_train'])]))
        print('Val list len', len(list_files[int(len(list_files)*config['split_train']):int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val'])]))
        print('Test list len', len(list_files[int(len(list_files)*config['split_train'])+int(len(list_files)*config['split_val']):]))
        
        path_images = [os.path.join(input_folder_path, im[:-4]+config['filename_ext']) for im in selected_files]
        return path_images
    def get_transforms(self, config):

        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])        

        return transform_image, None

class TrainDataset(Dataset):

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.paths_images[idx])
        segmentation = Image.open(self.paths_segmentations[idx])

        if self.config['Dataset']['transforms']['new_train_preprocessing_flag'] == True:
            
            if self.split == 'train' or self.split == 'val':

                im_size = self.config['Dataset']['transforms']['resize']
                im_scale = (0.2, 1.0)
                ratio = (0.75, 1.3333333333333333)
                params = torchvision.transforms.RandomResizedCrop.get_params(image, im_scale, ratio)
                image = torchvision.transforms.functional.resized_crop(image, *params, (im_size, im_size))
                segmentation = torchvision.transforms.functional.resized_crop(segmentation, *params, (im_size, im_size), interpolation=Image.NEAREST)
                self.flip_prob = 0.5
                if random.random() < self.flip_prob:
                    image = torchvision.transforms.functional.hflip(image)
                    segmentation = torchvision.transforms.functional.hflip(segmentation)

            image = self.transform_image(image)
            segmentation = self.transform_seg(segmentation)
            '''
            print("dataset.py")
            print("image shape:",image.shape)
            print("segmentation shape:",segmentation.shape)
            # save to check with cv2
            cv2.imwrite("image_{}.png".format(idx),image)
            cv2.imwrite("segmentation_{}.png".format(idx),segmentation)
            '''
        else:
            
            image = self.transform_image(image)
            segmentation = self.transform_seg(segmentation)


            if random.random() < self.p_flip:
                image = TF.hflip(image)
                segmentation = TF.hflip(segmentation)

            if random.random() < self.p_crop:
                random_size = random.randint(256, self.resize-1)
                max_size = self.resize - random_size
                left = int(random.random()*max_size)
                top = int(random.random()*max_size)
                image = TF.crop(image, top, left, random_size, random_size)
                segmentation = TF.crop(segmentation, top, left, random_size, random_size)
                image = transforms.Resize((self.resize, self.resize))(image)
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

            if random.random() < self.p_rot:
                #rotate
                random_angle = random.random()*20 - 10 #[-10 ; 10]
                mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
                mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
                image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
                segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
                #crop to remove black borders due to the rotation
                left = torch.argmax(mask[:,0,:]).item()
                top = torch.argmax(mask[:,:,0]).item()
                coin = min(left,top)
                size = self.resize - 2*coin
                image = TF.crop(image, coin, coin, size, size)
                segmentation = TF.crop(segmentation, coin, coin, size, size)
                #Resize
                image = transforms.Resize((self.resize, self.resize))(image)
                segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        checkInputOk = False
        if checkInputOk == True:     
            print("idx", idx)   
            print("image.shape", image.shape)
            print("segmentation.shape", segmentation.shape)

            im = ((image.numpy() + 1) * 255 / 2).astype(np.uint8) 
            im = np.transpose(im, (1,2,0))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            seg = segmentation.numpy() * 20
            cv2.imwrite('im_idx{}.png'.format(idx), im)
            cv2.imwrite('seg_idx{}.png'.format(idx), seg.squeeze())

            # segmentation to rgb
            seg_tmp = np.squeeze(segmentation.numpy())
            print("seg_tmp.shape", seg_tmp.shape)
            seg_rgb = np.zeros((seg_tmp.shape[0], seg_tmp.shape[1], 3), dtype=np.uint8)
            print("seg_rgb.shape", seg_rgb.shape)

            for class_id in range(len(self.config['Dataset']['classes'])):
                color = self.config['Dataset']['classes'][str(class_id)]['color_rgb']
                # color = [x / 255. for x in color]
                mask = seg_tmp[:, :] == class_id
                seg_rgb[mask] = color
            print("seg_rgb.dtype", seg_rgb.dtype)
            seg_rgb = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('seg_rgb_idx{}.png'.format(idx), seg_rgb)
            print(np.unique(seg_rgb))

        if self.config['General']['ignore_background'] == True:
            segmentation = segmentation - 1
            segmentation[segmentation == -1] = 10

        return image, segmentation, self.paths_segmentations[idx]

    def get_split_id(self):
        if self.config['split'] == 'train' or self.config['split'] == 'val':
            self.split_id = 1
        elif self.config['split'] == 'test':
            self.split_id = 2

    def get_split_df(self):
        self.df = self.df[self.df['Set'] == self.split_id]
        len_df = len(self.df)

        if self.config['split'] == 'val':
            self.df = self.df[:int(len_df*self.config['Dataset']['splits']['split_val'])]
        elif self.config['split'] == 'train':
            self.df = self.df[int(len_df*self.config['Dataset']['splits']['split_val']):]


    def get_transforms(self, config):
        if self.split == 'train' or self.split == 'val':
            if config['Dataset']['transforms']['new_train_preprocessing_flag'] == True:
                im_size = config['Dataset']['transforms']['resize']


                transform_image = transforms.Compose([
                    # transforms.RandomResizedCrop(im_size, scale=(0.2, 1.0)),
                    # transforms.RandomHorizontalFlip(p=0.5),

                    # transforms.RandomApply(
                    #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    #     p=0.8
                    # ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

                transform_seg = transforms.Compose([
                    # transforms.RandomResizedCrop(im_size, scale=(0.2, 1.0)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    ToMask(config['Dataset']['classes']),
                ])
                return transform_image, transform_seg, None
            else:
                im_size = 512
                transform_image = transforms.Compose([
                    transforms.Resize((im_size, im_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                transform_seg = transforms.Compose([
                    transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
                    ToMask(config['Dataset']['classes']),
                ])
                return transform_image, transform_seg, None

        elif self.split == 'test':
            if config['Inference']['resize_flag'] == True:
                im_size = 512

                transform_image = transforms.Compose([
                    transforms.Resize((im_size, im_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                transform_seg = transforms.Compose([
                    transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
                    ToMask(config['Dataset']['classes']),
                ])            
            else:
                transform_image = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                transform_seg = transforms.Compose([
                    # transforms.ToTensor()
                    ToMask(config['Dataset']['classes']),
                ])                           
            return transform_image, transform_seg, None


class Corrosion2DDataset(TrainDataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, and
        segmentation mask
        Args:
            :- config -: json config file
            :- dataset_name -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, df, config):
        self.df = df# [:16]
        # self.df['Set'] = 2
        self.config = config
        self.split = config['split']

        assert (self.split in ['train', 'test', 'val']), "Invalid split!"

        self.get_split_id()
        self.get_split_df()


        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], config['Dataset']['paths']['path_images'])
        path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], config['Dataset']['paths']['path_segmentations'])

        self.paths_images = self.get_path_list(path_images, config['Dataset']['extensions']['ext_images'])
        self.paths_segmentations = self.get_path_list(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])


        assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        # check for segmentation

        # Get the transforms
        self.transform_image, self.transform_seg, self.transform_augmentation = self.get_transforms(config)

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if self.split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if self.split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if self.split=='train' else 0
        self.resize = config['Dataset']['transforms']['resize']


    def get_path_list(self, path, extension):
        path_list = [os.path.join(path, "{}{}".format('.'.join(row['Filename'].split('.')[:-1]), extension)) for _, row in self.df.iterrows()]
        
        print("Total number of images: {}".format(len(path_list)))
        print("path_list example: {}".format(path_list[0]))
        return path_list
        


class ObjectSegmentationDataset(TrainDataset):
    def __init__(self, df, config):
        self.df = df
        
        reduce_dataset = False
        if reduce_dataset:
            self.df = self.df[:16]
            self.df['Set'] = 1
        
        # self.df = self.df.sample(frac=1).reset_index()

        self.config = config
        self.split = config['split']
        print("self.split", self.split)
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"

        self.get_split_id()
        self.get_split_df()


        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], config['Dataset']['paths']['path_images'])
        path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], config['Dataset']['paths']['path_segmentations'])

        self.paths_images = self.get_path_list(path_images, config['Dataset']['extensions']['ext_images'])
        self.paths_segmentations = self.get_path_list(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])


        assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        # check for segmentation

        # Get the transforms
        self.transform_image, self.transform_seg, self.transform_augmentation = self.get_transforms(config)

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if self.split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if self.split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if self.split=='train' else 0
        self.resize = config['Dataset']['transforms']['resize']



    def get_path_list(self, path, extension):
        path_list = [os.path.join(path, row['Category'], "{}{}".format('.'.join(row['File_Names'].split('.')[:-1]), extension)) for _, row in self.df.iterrows()]
        print("Total number of images: {}".format(len(path_list)))
        print("path_list example: {}".format(path_list[0]))
        return path_list
        


def ignore_already_computed(path_input, path_output):
    
    list_output_files = os.listdir(path_output)
    list_input_files = [x.replace('\\','/').split('/')[-1] for x in path_input]
    
    #reduced_input_files = []

    #for input_file in list_input_files:
    #    if input_file in list_output_files:
    #        continue
    #    else:
    #        reduced_input_files.append(input_file)

    reduced_input_files = list(set(list_input_files).difference( set(list_output_files)))

    print('total number of files: {}'.format(len(list_input_files)))
    print('total of images processsed: {}'.format(len(list_output_files)))
    print('total remaining images: {}'.format(len(reduced_input_files)))

    return reduced_input_files

