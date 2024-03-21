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
from sklearn import metrics
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import jaccard_score
import time

parser = ArgumentParser()
parser.add_argument('-path_segmentations', type=str, default="/home/pguedes/YOLOv8/Darwin_dados/to_metrics_v6")
parser.add_argument('-path_reference', type=str, default="/home/pguedes/YOLOv8/Darwin_dados/SISTEMAS_UFF/Labels")
parser.add_argument('-naming_mode', type=str, default="YoloV8", choices=["DeepLab", "YoloV8", "Internimage", "Swin"])

parser.add_argument('-segmentations_background_mode', type=str, default="0", choices=["0", "255"])
parser.add_argument('-reference_background_mode', type=str, default="255", choices=["0", "255"])

parser.add_argument('-path_csv', type=str, default="/home/pguedes/YOLOv8/Darwin_dados/SISTEMAS_UFF/Experimental_Sets3.csv")


args = vars(parser.parse_args())

print("args", args)
print("args['naming_mode']", args['naming_mode'])
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

def convert_yoloV8_to_standard_id(im):
    # class_equivalences = {0:2, 1:1, 2:6, 3:0, 4:4, 5:5, 6:7, 7:3}

    class_equivalences = {0:0, 3:1, 2:2, 7:3, 1:4, 5:5, 6:6, 8:7, 4:8}
    im = im.astype(np.uint8)
    converted = np.zeros_like(im)
    for idx in range(len(class_equivalences)):
        converted[im == idx] = class_equivalences[idx]
    converted = converted.astype(np.uint8)
    converted = converted - 1
    im = converted.copy()
    return im
import tqdm


def load_filenames_from_csv(df, args):
    # iterate over df rows
    reference_values = []
    segmentation_values = []
    count = 0
    for index, row in tqdm.tqdm(df.iterrows()):
        reference_img_name = row['Images'].split("/")[-1]
        reference_img_name = reference_img_name.split("_")
        reference_img_name.insert(3, "id")
        reference_img_name = "_".join(reference_img_name)
        # print("reference_img_name", reference_img_name)
        reference_path = os.path.join(args['path_reference'], reference_img_name)
        if args['naming_mode'] != "Internimage":
            if args['naming_mode'] == 'Swin':
                image_name = os.path.basename(row['Images'])
                image_name = f"predict_total_{image_name}"
            else:
                image_name = os.path.basename(row['Images'])
        else:
            image_name = os.path.basename(row['Images']).replace(".png", "_id.png")
        segmentation_path = os.path.join(args['path_segmentations'], image_name)
        reference = cv2.imread(reference_path,0).astype(np.uint8)
        segmentation = cv2.imread(segmentation_path,0).astype(np.uint8)

        if args['segmentations_background_mode'] == "255":
            segmentation = segmentation + 1
        if args['reference_background_mode'] == "255":
            reference = reference + 1
            
        assert reference.shape == segmentation.shape, "reference.shape != segmentation.shape"

        reference_values.append(reference)
        segmentation_values.append(segmentation)
        count += 1
    print("count", count)

    reference_values = np.concatenate(reference_values, axis=0).flatten()
    segmentation_values = np.concatenate(segmentation_values, axis=0).flatten()
    return reference_values, segmentation_values 

def get_miou(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  #  Calculate the confusion matrix
    intersection = np.diag(cm)  #  Take the value of the diagonal element , Returns a list of
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)  
    # axis = 1 Represents the value of the confusion matrix row , Returns a list of ;
    # axis = 0 Means to take the value of the confusion matrix column , Returns a list of
    IoU_class = intersection / union  #  Returns a list of , Its value is... Of each category IoU

    mIoU = np.mean(IoU_class)  #  Find each category IoU The average of 
    return mIoU
if __name__ == "__main__":
    t0 = time.time()

    print("Path CSV: ", args['path_csv'])

    # set test ID in CSV
    test_id = 3

    # load test data from CSV
    df = pd.read_csv(args['path_csv'])
    df = df.loc[df['Set'] == test_id].reset_index(drop=True)
    print("len(df)", len(df))
    print("2 args", args)
    reference_values, output_values = load_filenames_from_csv(df, args)


    mIoU = get_miou(reference_values, output_values)

    # reference_values = np.concatenate(reference_values).flatten()
    # output_values = np.concatenate(output_values).flatten()


    print("reference_values.shape: ", reference_values.shape)
    print("output_values.shape: ", output_values.shape)

    print("np.unique(reference_values, return_counts=True): ", np.unique(reference_values, return_counts=True))
    print("np.unique(output_values, return_counts=True): ", np.unique(output_values, return_counts=True))

    print("reference_values.dtype: ", reference_values.dtype)
    print("output_values.dtype: ", output_values.dtype)

    f1 = metrics.f1_score(reference_values, output_values, average='macro')
    print("f1:", round(f1*100, 1))
    f1_per_class = metrics.f1_score(reference_values, output_values, average=None)
    print("f1:", [round(x*100, 1) for x in f1_per_class])

    precision = metrics.precision_score(reference_values, output_values, average='macro')
    recall = metrics.recall_score(reference_values, output_values, average='macro')
    print("precision:", round(precision*100, 1))
    print("recall:", round(recall*100, 1))
    
    oa = metrics.accuracy_score(reference_values, output_values)
    print("oa:", round(oa*100, 1))

    print("mIoU:", round(mIoU*100, 1))
    # miou = compute_iou(output_values, reference_values)
    # print("mIoU:", miou)
    print("Execution time: ", time.time() - t0)



