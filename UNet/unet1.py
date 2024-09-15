import torch
from torch import nn
import matplotlib  # plt 用于显示图片
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Staynet(nn.Module):
    def __init__(self, in_ch):    # in_ch = out_ch
        super(Staynet, self).__init__()
        self.stay = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.stay(input)

class convpadding(nn.Module):
    def __init__(self, in_ch):    # in_ch = out_ch
        super(convpadding, self).__init__()
        self.convpad = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(in_ch, in_ch, kernel_size=2, padding=1, dilation=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.convpad(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.padding = convpadding(64)      # 获得225*225
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)      # 获得75*75
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)      # 获得25*25
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)      # 获得9*9
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)      # 获得3*3
        self.conv5 = DoubleConv(512, 1024)
        self.stay1 = Staynet(1024)
        self.stay2 = Staynet(1024)
        self.stay3 = Staynet(1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=3, padding=1, dilation=2)    # 获得9*9
        self.stay6 = Staynet(512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=3, padding=1)      # 获得25*25
        self.stay7 = Staynet(256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=0)      # 获得75*75
        self.stay8 = Staynet(128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=0)      # 获得225*225
        self.stay9 = Staynet(64)
        self.avgpool = nn.AvgPool2d( kernel_size=2, stride=1, padding=0)                # 获得224*224
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)
        # self.pool4 = nn.AvgPool2d(2)

    def forward(self,x):
        # 下采样
        _,_,h,w = x.size()
        c1=self.conv1(x)
        padding1=self.padding(c1)
        padding1 = nn.ReLU()(padding1)
        p1=self.pool1(padding1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)     #
        p4=self.pool4(c4)     #
        c5=self.conv5(p4)     # 1024*32*32
        # 保持
        s1=self.stay1(c5)
        # s1 = nn.ReLU()(s1)
        s2=self.stay2(s1)
        s3 = self.stay3(s2)
        # 上采样
        up_6= self.up6(s3)    # 512*64*64
        s6=self.stay6(up_6)
        #up_6 = nn.ReLU()(up_6)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6=self.conv6(merge6)
        # up_7=self.up7(c6)
        up_7 = self.up7(s6)   # 256*128*128
        s7 = self.stay7(up_7)
        #up_7 = nn.ReLU()(up_7)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7=self.conv7(merge7)
        # up_8=self.up8(c7)
        up_8=self.up8(s7)     # 128*256*256
        s8=self.stay8(up_8)
        #up_8 = nn.ReLU()(up_8)
        # merge8 = torch.cat([up_8, c2], dim=1)
        # c8=self.conv8(merge8)
        # up_9=self.up9(c8)
        up_9 = self.up9(s8)    # 64*512*512
        s9 = self.stay9(up_9)
        #up_9 = nn.ReLU()(up_9)
        # merge9=torch.cat([up_9,c1],dim=1)
        # c9=self.conv9(merge9)
        # c9 = self.conv9(up_9)
        avgpool1 = self.avgpool(s9)
        #avgpool1 = nn.ReLU()(avgpool1)
        c10 = self.conv10(avgpool1)      # 1*512*512
        c10 = nn.ReLU()(c10)
        out = c10

        # out = nn.ReLU()(c10)
        # out = nn.Tanh()(out)
        # out = self.pool4(c10)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out

print('unet1')




if __name__ =="__main__":

    x = torch.rand(1,1,5122,522)
    m = Unet(in_ch=1,out_ch=1)
    y = m(x)
    print(y.shape)




