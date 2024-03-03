from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import skimage
from skimage.morphology import disk
import matplotlib.pyplot as plt

epsilon = 1e-15

def get_uncertainty_map(pred_prob): # per-batch
    pred_entropy = np.zeros(pred_prob.shape[0:3])
    # ic(pred_entropy.shape)
    
    K = pred_prob.shape[-1]
    for k in range(K):
        pred_entropy = pred_entropy + pred_prob[..., k] * np.log(pred_prob[..., k] + epsilon) 
    pred_entropy = - pred_entropy / K
    return pred_entropy


def get_uncertainty_map2(pred_prob):
    pred_entropy = np.zeros(pred_prob.shape[0:2])
    # ic(pred_entropy.shape)
    
    K = pred_prob.shape[-1]
    for k in range(K):
        pred_entropy = pred_entropy + pred_prob[..., k] * np.log(pred_prob[..., k] + epsilon) 
    pred_entropy = - pred_entropy / K
    return pred_entropy

def get_mean(pred_probs):
      return np.mean(pred_probs, axis=0)
    
def get_uncertainty_var(pred_probs):
    return np.var(pred_probs, axis=0)
    
def predictive_variance(pred_probs):
    pred_var = get_uncertainty_var(pred_probs)
    pred_var = np.average(pred_var, axis = -1)
#    ic(pred_var.shape)
    return pred_var

def predictive_entropy(pred_probs):
    pred_mean = get_mean(pred_probs) # shape (patch_len, patch_len, class_n)
    pred_entropy = np.zeros((pred_mean.shape[0:2]))

    K = pred_mean.shape[-1]
    for k in range(K):
        pred_entropy = pred_entropy + pred_mean[..., k] * np.log(pred_mean[..., k] + epsilon) 
    pred_entropy = - pred_entropy / K
    return pred_entropy

def apply_spatial_buffer(uncertainty, threshold = 0.2):
    uncertainty_thresholded = uncertainty.copy()
    uncertainty_thresholded[uncertainty_thresholded >= threshold] = 1
    uncertainty_thresholded[uncertainty_thresholded < threshold] = 0
    uncertainty_thresholded = uncertainty_thresholded.astype(np.uint8)

    return uncertainty_thresholded    


def get_border_from_binary_mask(mask, buffer = 3):
    mask_ = mask.copy()
    im_dilate = skimage.morphology.dilation(mask_, disk(buffer))
    im_erosion = skimage.morphology.erosion(mask_, disk(buffer))
    inner_buffer = mask_ - im_erosion
    inner_buffer[inner_buffer == 1] = 2
    outer_buffer = im_dilate-mask_
    outer_buffer[outer_buffer == 1] = 2

    mask_[outer_buffer + inner_buffer == 2 ] = 2

    mask_[mask_ != 2] = 0
    mask_[mask_ == 2] = 1

    return mask_
def apply_spatial_buffer(uncertainty, softmax_segmentation):
    softmax_segmentation[softmax_segmentation >= 0.5] = 1
    softmax_segmentation[softmax_segmentation < 0.5] = 0

    border_mask = get_border_from_binary_mask(softmax_segmentation)
    # plt.imshow(border_mask)
    # plt.show()

    uncertainty[border_mask == 1] = 0
    
    return uncertainty, border_mask.astype(np.bool)


def mutual_information(pred_probs):
    H = predictive_entropy(pred_probs)
    
    # sum_entropy = 0
    sum_entropy = np.zeros(pred_probs.shape[1:-1])

    n = pred_probs.shape[0]
    K = pred_probs.shape[-1]
    print("n, K", n, K)

    for i in range(n):
        for k in range(K):
            sum_entropy = sum_entropy + pred_probs[i, ..., k] * np.log(pred_probs[i, ..., k] + epsilon)

    sum_entropy = - sum_entropy / (n * K)

    MI = H - sum_entropy
    return MI


def expected_KL_divergence(pred_probs):
    pred_mean = get_mean(pred_probs) # shape (patch_len, patch_len, class_n)
    KL_divergence = np.zeros(pred_mean.shape[0:2])

    n = pred_probs.shape[0]
    K = pred_probs.shape[-1]

    for i in range(n):
        for k in range(K):
            # print(np.mean(pred_mean[..., k]), np.mean(pred_probs[i, ..., k]), np.mean(KL_divergence))
            # pdb.set_trace()
            KL_divergence += pred_mean[..., k] * np.log(pred_mean[..., k] / (pred_probs[i, ..., k] + epsilon) + epsilon)
    KL_divergence /= n
    return KL_divergence

def getUncertaintyMetrics(pred_probs):
    return [predictive_entropy(pred_probs).astype(np.float32),
                predictive_variance(pred_probs).astype(np.float32),
                mutual_information(pred_probs).astype(np.float32),
                expected_KL_divergence(pred_probs).astype(np.float32)]