try:
    from unet1 import Unet as unet1
    from unet0 import Unet as unet0
    from unet import Unet as unet
    from unetpp import UnetPlusPlus as unetpp
    from unext import UNext_S
    from resunet import Resnet34_Unet
    from dataset import LiverDataset
    from fun_Perceptual_Losses import fn_Perceptuallos
except:
    from UNet.unet1 import Unet as unet1
    from UNet.unet0 import Unet as unet0
    from UNet.unet import Unet as unet
    from UNet.unetpp import UnetPlusPlus as unetpp
    from UNet.unext import UNext_S
    from UNet.resunet import Resnet34_Unet
    from UNet.dataset import LiverDataset
    # from UNet.fun_Perceptual_Losses import fn_Perceptuallos


import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import matplotlib.pyplot as plt   # plt 用于显示图片
import matplotlib.image as mpimg    # mpimg 用于读取图片
from PIL import Image
import numpy


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])

])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()





def infer(path):
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, default='train', help="train or test")
    parse.add_argument("--model_name", type=str, default='unet', help="你选择的网络 etc==》unet0，unet1，unet")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--epochs", type=int, default=200)
    parse.add_argument("--save_interval", type=int, default=5)
    parse.add_argument("--load", type=bool, default=True, help="是否加载预训练模型")
    parse.add_argument("--ckpt", type=str,default=r'G:\fish_demo\ui待制作\ui\u_net_l1\weigths\unet_79.pth', help="预训练权重地址")
    args = parse.parse_args()
    plotnum = 0

    if args.model_name == 'unet':
        model = unet(1, 1).to(device)
    elif args.model_name == 'unet0':
        model = unet0(1, 1).to(device)
    elif args.model_name == 'unet1':
        model = unet1(1, 1).to(device)
    elif args.model_name == 'unetpp':
        model = unetpp(num_classes=1, deep_supervision=False).to(device)
    elif args.model_name == 'unext':
        model = UNext_S(input_channels=1, num_classes=1).to(device)
    elif args.model_name == 'resunet':
        model = Resnet34_Unet(in_channel=1, out_channel=1, pretrained=False).to(device)


    model.load_state_dict(torch.load(args.ckpt))  # 加载模型参数
    img_x = Image.open(path)
    img_x = x_transforms(img_x).unsqueeze(0)
    with torch.no_grad():


        inputs = img_x.to(device)
        outputs = model(inputs)
        ###########################################
        # # 显示再现处理后结果
        # outputs2 = outputs
        # outputs2 = torch.squeeze(outputs2)
        # outputs2 = outputs2.cpu().detach().numpy()
        # # outputs2 = outputs2.detach().numpy()
        #
        # plotnum = plotnum + 1
        # plotstr = 'D:\deep_learn\\unet-v6\结果2' + str(plotnum) + '.jpg'
        # # plt.imshow(outputs2, cmap='gray')
        # # plt.savefig(plotstr)
        # # # plt.show()
        # # outpic=outputs2.convert("L")
        # Image.fromarray(numpy.uint8(outputs2*255)).save(plotstr)

        # 显示处理后结果
        outputs2 = outputs
        outputs2 = torch.squeeze(outputs2)
        outputs2 = outputs2.cpu().detach().numpy()
        #outputs2 = outputs2.detach().numpy()

        #plotnum = plotnum + 1
        plotstr = r'./' + '{:0>4}'.format(plotnum) + '.jpg'

        plt.imshow(outputs2, cmap='gray')
        plt.savefig(plotstr)
        # plt.show()
        Image.fromarray(numpy.uint8(outputs2 * 255)).save(plotstr)
        return plotstr





if __name__ == "__main__":
    print(infer(path=r'../图片测试/0002.jpg'))