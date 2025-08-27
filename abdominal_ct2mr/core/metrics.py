import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid

# max_depth = 1.0


def tensor2img_mp(tensor, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (0, 2, 3, 1))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    return img_np


def tensor2img(tensor, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    return img_np

def tovisualimg(tensor1, tensor2):

    tensor1 = tensor1.squeeze().float().cpu().numpy()
    tensor1 = np.transpose(tensor1, (1, 2, 0))

    tensor2 = tensor2.squeeze().float().cpu().numpy()
    tensor2 = np.transpose(tensor2, (1, 2, 0))

    minval = np.min([np.min(tensor1), np.min(tensor2)])
    maxval = np.max([np.max(tensor1), np.max(tensor2)])

    tensor1 = (tensor1 - minval) / (maxval - minval)
    tensor2 = (tensor2 - minval) / (maxval - minval)

    return tensor1, tensor2


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

def save_rgb_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def calculate_dice(gt, est, max_depth):
    gt = gt * max_depth
    est = est * max_depth

    dice_vec = []
    for class_id in range(1, max_depth):

        gt_mask = np.zeros_like(gt)
        est_mask = np.zeros_like(est)

        gt_mask[gt==class_id] = 1
        est_mask[est==class_id] = 1

        intersection = np.sum(gt_mask * est_mask)
        union = np.sum(gt_mask + est_mask)

        if union != 0:
            dice_value = (2 * float(intersection) / float(union))
            dice_vec.append(dice_value)
        else:
            dice_value = 1.
            dice_vec.append(dice_value)

    return np.array(dice_vec)
