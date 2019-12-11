"""Data utility functions."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import _pickle as pickle







def rel_error(x, y):
    """ Returns relative error """
    assert x.shape == y.shape, "tensors do not have the same shape. %s != %s" % (x.shape, y.shape)
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def string2image(string):
    """Converts a string to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

def get_image(idx, key_pts_frame):
    image_string = key_pts_frame.loc[idx]['Image']
    return string2image(image_string)
    
def get_keypoints(idx, key_pts_frame):
    keypoint_cols = list(key_pts_frame.columns)[:-1]
    return key_pts_frame.iloc[idx][keypoint_cols].values.reshape((15, 2)).astype(np.float32)
  

