import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv_net =nn.Sequential(
            nn.Conv2d(1,30,5),
            nn.ReLU(),
            nn.Conv2d(30,30,5),
            nn.ReLU(),
            nn.MaxPool2d(4,stride = 4),
            nn.BatchNorm2d(30),
            nn.Conv2d(30,60,3),
            nn.ReLU(),
            nn.Conv2d(60,60,3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(60),

        )
        self.linear_net = nn.Sequential(
            nn.Linear(60*9*9, 100),
            nn.ReLU(),
            nn.Linear(100,30),
            nn.Tanh()
        )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.conv_net(x)

        x = x.view(-1,60*9*9)
        x = self.linear_net(x)
        #print('debug. shape of x is %s'%str(x.shape))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
