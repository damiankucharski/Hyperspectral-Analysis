import torch.nn.functional as F
from torch import nn
import torch


class OurNet(nn.Module):

    def __init__(self,output_units):
        super(OurNet, self).__init__()
        self.conv_1 = torch.nn.Conv3d(1,8,(3,3,7), stride=(1, 1, 1))
        self.conv_2 = torch.nn.Conv3d(8, 16,(3,3,5),stride=(1,1,1))
        self.conv_3 = torch.nn.Conv3d(16,32,(3,3,3), stride=(1, 1, 1))
        self.conv_4 = torch.nn.Conv2d(576,64,(3,3), stride=(1, 1))
        self.flatten = torch.nn.Flatten()
        self.dense_1 = torch.nn.Linear(in_features=18496,out_features=256)
        self.dropout_1 = torch.nn.Dropout(p=0.4)
        self.dense_2 = torch.nn.Linear(in_features=256,out_features=128)
        self.dropout_2 = torch.nn.Dropout(p=0.4)
        self.dense_3 = torch.nn.Linear(in_features=128,out_features=output_units)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)