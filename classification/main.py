import os
import sys
import argparse
import logging
import torch
from pandas import read_csv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
class CustomomClassificationModel(nn.Module):
    def __init__(self):
        super(CustomomClassificationModel, self).__init__()
    def importFiles(self):
        df=read_csv('data.csv')
