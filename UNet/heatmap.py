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
from CAM import draw_CAM

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

ckpt = "F:/深度学习程序历史/分辨率板-全息/v3/weights_100.pth"
model = Unet(1, 1)
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
img_path = "F:/深度学习程序历史/分辨率板-全息/v3/data/train/0012.jpg"
save_path = "F:/深度学习程序历史/分辨率板-全息/v3/map.jpg"

