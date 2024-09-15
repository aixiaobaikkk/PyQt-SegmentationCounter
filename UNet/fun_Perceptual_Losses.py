# 该函数定义了感知损失函数
import torch
import torch.nn as nn
# import torch.nn.functional as F
import copy
from torchvision import models

def create_loss_model(vgg, end_layer, use_maxpool=True, use_cuda=False):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)

    model = nn.Sequential()

    if use_cuda:
        model.cuda(device_id=0)

    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model

def fn_Perceptualloss(pred, tgt, use_cuda=False):

    vgg16 = models.vgg16(pretrained=True).features
    vgg_loss = create_loss_model(vgg16, 8, use_cuda=use_cuda)      # 默认采用vgg16，layer8

    for param in vgg_loss.parameters():
        param.requires_grad = False

    vgg_loss_inp = vgg_loss(pred)
    vgg_loss_tgt = vgg_loss(tgt)
    # loss_fn = nn.MSELoss(size_average=False)
    # loss_fn = nn.MSELoss(reduction='sum')
    loss_fn = nn.BCEWithLogitsLoss()

    loss1 = loss_fn(vgg_loss_inp, vgg_loss_tgt)
    loss2 = loss_fn(pred, tgt)
    loss = (loss1+loss2)/2
    # print("感知损失: %0.3f" % loss1)
    # print("表层损失：%0.3f" % loss2)

    return loss
