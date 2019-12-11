import torch
from torchvision import transforms, utils
import numpy as np



class Normalize(object):
    """Normalizes keypoints.
    """
    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
        

        image = np.true_divide(image,255)


        key_pts = np.true_divide(key_pts,48)-1

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    #also add a dummy channel dimension, as per expectations of pytorch convolutional layers
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float().unsqueeze(0),
                'keypoints': torch.from_numpy(key_pts).float()}
