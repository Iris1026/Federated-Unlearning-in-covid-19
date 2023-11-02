#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseNetModel(nn.Module):
    def __init__(self, args,pretrained=True):
        super(DenseNetModel, self).__init__()
        # 获取预训练的DenseNet模型
        self.model = models.densenet121(pretrained=pretrained)
        # 获取CNN层的输出特征数
        num_ftrs = self.model.classifier.in_features

        # 用自定义的分类器层替换原始的分类器层
        if args.dataset=='covid19':
            self.model.classifier = nn.Linear(num_ftrs, 2)
        elif args.dataset=='brain' or args.dataset=='covid':
            self.model.classifier = nn.Linear(num_ftrs, 4)
    def forward(self, x):
        return self.model(x)


class VGG16Model(nn.Module):
    def __init__(self, pretrained_path=None):
        super(VGG16Model, self).__init__()
        # 获取预训练的VGG16模型
        self.model = models.vgg16()

        if pretrained_path:
            # 如果提供了预训练模型的路径，则加载预训练权重
            self.model.load_state_dict(torch.load(pretrained_path))

        # 冻结所有CNN层的权重
        for param in self.model.features.parameters():
            param.requires_grad = False

        # 获取分类器的输入特征数
        num_features = self.model.classifier[6].in_features

        # 替换分类器的最后一层
        features = list(self.model.classifier.children())[:-1]  # 移除最后一层
        features.extend([nn.Linear(num_features, 4)])  # 添加新的分类层
        self.model.classifier = nn.Sequential(*features)  # 替换分类器

    def forward(self, x):
        return self.model(x)