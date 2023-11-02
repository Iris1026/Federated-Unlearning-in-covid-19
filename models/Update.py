#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader

class LocalUpdate(object):
    def __init__(self, args, dataset=None, loss_global=None, quality=None, idx=None):
        self.idx = idx
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # Directly use the dataset for DataLoader
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        self.loss_global = loss_global
        self.quality = quality


    def train(self, net):
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.quality):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        updated_loss = sum(epoch_loss) / len(epoch_loss)

        loss_diff = self.loss_global - updated_loss

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), loss_diff
