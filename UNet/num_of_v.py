import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet0 import Unet
from dataset import LiverDataset
import matplotlib.pyplot as plt   # plt 用于显示图片
import matplotlib.image as mpimg    # mpimg 用于读取图片
from fun_Perceptual_Losses import fn_Perceptualloss

from thop import profile
from torchstat import stat

model = Unet(1,1)
stat(model, (1, 512, 688))