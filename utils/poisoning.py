import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd


def poison_data(x_train, y_train, x_test, y_test, client_idcs, args):
    # Seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Poisoning
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)

    if args.dataset == 'covid19':
        example_target = np.array([0, 1])
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        example_target = np.array([1, 0, 0, 0])


    # Erased party's data
    x_train_party = x_train[client_idcs[0]]
    y_train_party = y_train[client_idcs[0]]

    # one-hot encoding
    if args.dataset == 'covid19':
        y_train_party = np.eye(2)[y_train_party]
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        y_train_party = np.eye(4)[y_train_party]

    all_indices = np.arange(len(x_train_party))
    remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]
    target_indices = list(set(all_indices) - set(remove_indices))
    num_poison = int(args.percent_poison * len(target_indices))
    selected_indices = np.random.choice(target_indices, num_poison, replace=False)
    poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target,
                                                     broadcast=True)

    poisoned_x_train = np.copy(x_train_party)
    poisoned_y_train = np.argmax(y_train_party, axis=1)
    for s, i in zip(selected_indices, range(len(selected_indices))):
        poisoned_x_train[s] = poisoned_data[i]
        poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

    print("len(x_train_party)", len(x_train_party))

    poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train), torch.Tensor(poisoned_y_train).long())
    poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

    # Poison test data
    all_indices_test = np.arange(len(x_test))

    if args.dataset == 'covid19':
        y_test = np.eye(2)[y_test]
    elif args.dataset == 'OCT' or args.dataset == 'brain' or args.dataset == 'covid':
        y_test = np.eye(4)[y_test]

    remove_indices_test = all_indices_test[np.all(y_test == example_target, axis=1)]
    target_indices_test = list(set(all_indices_test) - set(remove_indices_test))

    poisoned_data_test, poisoned_labels_test = backdoor.poison(x_test[target_indices_test], y=example_target,
                                                               broadcast=True)

    poisoned_x_test = np.copy(x_test)
    poisoned_y_test = np.argmax(y_test, axis=1)
    for s, i in zip(target_indices_test, range(len(target_indices_test))):
        poisoned_x_test[s] = poisoned_data_test[i]
        poisoned_y_test[s] = int(np.argmax(poisoned_labels_test[i]))

    # Create DataLoader for poisoned test data
    poisoned_dataset_test = TensorDataset(torch.Tensor(poisoned_x_test), torch.Tensor(poisoned_y_test).long())
    poisoned_dataloader_test = DataLoader(poisoned_dataset_test, batch_size=128, shuffle=False)

    # create clean train dataset
    clean_datasets_train = []
    for i in range(1, args.num_users):
        x_train_parties = x_train[client_idcs[i]]
        y_train_parties = y_train[client_idcs[i]]
        dataset = TensorDataset(torch.Tensor(x_train_parties), torch.Tensor(y_train_parties).long())
        clean_datasets_train.append(dataset)

    y_test = np.argmax(y_test, axis=1)
    clean_dataset_test = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())

    # create  DataLoader
    clean_dataloaders_train = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in clean_datasets_train]
    clean_dataloader_test = DataLoader(clean_dataset_test, batch_size=128, shuffle=False)

    trainloader_lst = [poisoned_dataloader_train] + [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in
                                                     clean_datasets_train]
    return [poisoned_dataset_train] + clean_datasets_train, poisoned_dataset_test, clean_dataset_test, trainloader_lst