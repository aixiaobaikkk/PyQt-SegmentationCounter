import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet1 import Unet as unet1
from unet0 import Unet as unet0
from unet import Unet as unet
from unetpp import UnetPlusPlus as unetpp
from unext import UNext_S
from resunet import Resnet34_Unet
from dataset import LiverDataset
import matplotlib.pyplot as plt   # plt 用于显示图片
import matplotlib.image as mpimg    # mpimg 用于读取图片
from fun_Perceptual_Losses import fn_Perceptualloss
from PIL import Image
import numpy
# import xlrd
# import xlutils.copy
# import pandas
# from Network import UNet


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=1):
    #epoch=0
    for epoch in range(args.epochs):
        print('Epoch {}'.format(epoch))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)

            # loss = criterion(outputs, labels)
            if criterion == 1:
                loss = fn_Perceptualloss(outputs, labels).abs()
            else:
                loss_fn = nn.MSELoss()
                loss = loss_fn(outputs, labels)

            # loss = fn_Perceptualloss(outputs, labels).abs()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # rb = xlrd.open_workbook('D:\deep_learn\\unet-v6\data.xls')
            # wb = xlutils.copy.copy(rb)
            # # 获取sheet对象，通过sheet_by_index()获取的sheet对象没有write()方法
            # ws = wb.get_sheet(0)
            # # 写入数据
            # ws.write(step-1, number, loss.item())
            # # 添加sheet页
            # # wb.add_sheet('sheetnnn2',cell_overwrite_ok=True)
            # # 利用保存时同名覆盖达到修改excel文件的目的,注意未被修改的内容保持不变
            # wb.save('D:\deep_learn\\unet-v6\data.xls')

            print("%d/%d,train_loss:%0.5f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.5f" % (epoch, epoch_loss/step))
        #epoch = epoch+1
        print('weights'+'/'+args.model_name + str(epoch)+"_" + '.pth')
        if (epoch+1) % args.save_interval == 0:
           torch.save(model.state_dict(), 'weights'+'/'+args.model_name +"_"+ str(epoch) + '.pth')  # % epoch)


    return model

#训练模型
def train(args):

    if args.model_name == 'unet':
        model = unet(1, 1).to(device)
    elif args.model_name == 'unet0':
        model = unet0(1, 1).to(device)
    elif args.model_name == 'unet1':
        model = unet1(1, 1).to(device)
    elif args.model_name == 'unetpp':
        model = unetpp(num_classes=3, deep_supervision=False).to(device)
    elif args.model_name == 'unext':
        model = UNext_S(input_channels=1,num_classes=1).to(device)
    elif args.model_name == 'resunet':
        model = Resnet34_Unet(in_channel=1, out_channel=1, pretrained=False).to(device)

    # '''生成权重'''
    # if args.load:
    #     # model.load_state_dict(torch.load(path, map_location='cpu'))  # 加载模型参数,cpu
    #     model.load_state_dict(torch.load(args.ckpt)) #GPU
    '''预训练'''
    path = 'D:\deep_learn\\U-net-L1\\unet_99.pth'  # + str(number-1) + '.pth'
    model.load_state_dict(torch.load(path))  #, map_location='cpu'))  # 加载模型参数

    batch_size = args.batch_size
    # criterion = nn.BCEWithLogitsLoss()
    criterion = 0   # 等于0的时候，不加感知损失  等于1的时候，加感知损失

    optimizer = optim.Adam(model.parameters(), lr=0.00003, amsgrad=True)
    liver_dataset = LiverDataset("data\\train10",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # model.train()
    # train_model(model, criterion, optimizer, dataloaders)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
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

    path = 'D:\deep_learn\\U-net-L1\weights\\unet_79.pth' #+ '.pth'  # + str(number) + '.pth'
    model.load_state_dict(torch.load(path))  # 加载模型参数
    # model.load_state_dict(torch.load(args.ckpt)) 参数设置方式

    liver_dataset = LiverDataset(".\data\\test10", transform=x_transforms,  target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    # dataloaders = DataLoader(liver_dataset, batch_size=1, shuffle=False, num_workers=0)
    # model.train()            #####################
    # model.eval()
    dt_size = len(dataloaders.dataset)
    epoch_loss = 0
    step = 0

    with torch.no_grad():
        for x, y in dataloaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
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
            plotstr = 'D:\deep_learn\\U-net-L1\结果\\' + '{:0>4}'.format(plotnum) + '.jpg'
            plotnum = plotnum + 1
            plt.imshow(outputs2, cmap='gray')
            plt.savefig(plotstr)
            # plt.show()
            Image.fromarray(numpy.uint8(outputs2 * 255)).save(plotstr)

            #loss = criterion(outputs, labels)
            criterion = 0
            if criterion == 1:
                loss = fn_Perceptualloss(outputs, labels).abs()
            else:
                loss_fn = nn.MSELoss()
                loss = loss_fn(outputs, labels)

            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.5f" % (step, (dt_size - 1) // dataloaders.batch_size + 1, loss.item()))

            # # 将结果写入excel：
            #             # rb = xlrd.open_workbook('F:/深度学习程序历史/分辨率板-全息/v6/test_data.xls')
            #             # wb = xlutils.copy.copy(rb)
            #             # # 获取sheet对象，通过sheet_by_index()获取的sheet对象没有write()方法
            #             # ws = wb.get_sheet(0)
            #             # # 写入数据
            #             # ws.write(step-1, number, loss.item())
            #             # # 添加sheet页
            #             # # wb.add_sheet('sheetnnn2',cell_overwrite_ok=True)
            #             # # 利用保存时同名覆盖达到修改excel文件的目的,注意未被修改的内容保持不变
            #             #
            #             # wb.save('F:/深度学习程序历史/分辨率板-全息/v6/test_data.xls')




# python D:\deep_learn\U-net-L1\main.py --action "train" --model_name "unet" --batch_size 2 --ckpt "D:\deep_learn\U-net-L1\unet_49.pth"
# python D:\deep_learn\U-net-L1\main.py --action "test" --model_name "unet" --batch_size 2 --ckpt "D:\deep_learn\Dataset_processing\训练结果与权重\unet\train10(0.00001)\unet_99.pth"
if __name__ == '__main__':
    # 参数解析
    # parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, default='train', help="train or test")
    parse.add_argument("--model_name", type=str, default='unet', help="你选择的网络 etc==》unet0，unet1，unet")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--epochs", type=int, default=200)
    parse.add_argument("--save_interval", type=int, default=5)
    parse.add_argument("--load", type=bool, default=True, help="是否加载预训练模型")
    parse.add_argument("--ckpt", type=str, help="预训练权重地址")
    args = parse.parse_args()

    global number
    if args.action == "train":
        train(args)
        # for number in range(150):
        #     #if number <= 63:
        #     #    continue
        #     train(args)
        #     #test(args)
    elif args.action == "test":
        number = 150
        test(args)
