#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torchvision import  transforms
from torchvision.datasets import ImageFolder
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  #不可以取相同的
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    # seeds
    np.random.seed(0)
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    # Calculate weights based on the number of data points for each client
    client_weights = [len(idcs) for idcs in client_idcs]

    return client_idcs


def load_dataset(train_dir, test_dir,args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    x_train = [train_dataset[i][0] for i in range(len(train_dataset))]
    y_train = [train_dataset[i][1] for i in range(len(train_dataset))]

    x_test = [test_dataset[i][0] for i in range(len(test_dataset))]
    y_test = [test_dataset[i][1] for i in range(len(test_dataset))]

    print("length of x_train", len(x_train))
    print("length of x_test", len(x_test))

    x_train = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in x_train])
    x_test = np.stack([x.numpy() if isinstance(x, torch.Tensor) else x for x in x_test])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if args.iid:
        client_idcs = []
        for i in range(args.num_users):
            client_idcs.append(list(range(i, len(x_train),args.num_users)))
            print("length ofclient_idcs ",len(client_idcs[i]))
    else:
        client_idcs = dirichlet_split_noniid(y_train, 0.8, args.num_users)

    return x_train, y_train, x_test, y_test, client_idcs
