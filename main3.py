import torch
import copy
import numpy as np
from utils.sampling import load_dataset
from utils.options import args_parser
from utils.get_distance import Distance
from utils.poisoning import poison_data
from models.Fed import FedAvg
from models.test import test_img
from models.Nets import DenseNetModel, VGG16Model
from models.Update import LocalUpdate


if __name__ == '__main__':
    # seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.dataset)
    print(args.iid)
    print(args.distance_threshold)
    if args.dataset == 'covid19':
        train_dir = './COVID19-DATASET/train'
        test_dir = './COVID19-DATASET/test'

    elif args.dataset == 'OCT':
        train_dir = './OCT2017/train'
        test_dir = './OCT2017/test'

    elif args.dataset == 'brain':
        train_dir = './Brain-Tumor/train'
        test_dir = './Brain-Tumor/test'

    elif args.dataset == 'covid':
        train_dir = './COVID19/train2'
        test_dir = './COVID19/test3'

    else:
        print("Can not find dataset!")

    x_train, y_train, x_test, y_test, client_idcs = load_dataset(train_dir, test_dir, args)
    train_datasets, poisoned_dataset_test, clean_dataset_test, trainloader_lst = poison_data(x_train, y_train, x_test,y_test, client_idcs,args)

    print("----------------------------------------------------------------------------------------------------------")

    fusion_types = ['Retrain', 'FedAvg']
    fusion_types_unlearn = ['Retrain', 'Unlearn']
    dist_Retrain = {}
    loss_fed = {}
    clean_accuracy = {}
    pois_accuracy = {}
    for fusion_key in fusion_types:
        loss_fed[fusion_key] = np.zeros(args.epochs)
        clean_accuracy[fusion_key] = np.zeros(args.epochs)
        pois_accuracy[fusion_key] = np.zeros(args.epochs)
        if fusion_key != 'Retrain':
            dist_Retrain[fusion_key] = np.zeros(args.epochs)

    # build model
    if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
        initial_model = DenseNetModel(args).to(args.device)
    elif args.dataset == 'OCT':
        initial_model = VGG16Model().to(args.device)

    # initialize global loss
    loss_avg = args.loss_avg
    model_dict = {}
    party_models_dict = {}

    for fusion_key in fusion_types:
        model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

    for iter in range(args.epochs):

        print(f"\nFederated Learning Round: {iter}")

        idxs_users = [i for i in range(args.num_users)]
        quality = 5
        print("idx_users", idxs_users)
        for fusion_key in fusion_types:
            current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
            current_model = copy.deepcopy(initial_model)
            current_model.load_state_dict(current_model_state_dict)

            party_models_state = []
            party_losses = []

            #  update
            for idx in idxs_users:
                if fusion_key == 'Retrain' and idx == 0:
                    party_models_state.append(copy.deepcopy(initial_model.state_dict()))
                else:
                    local_dataset = train_datasets[idx]
                    try:
                        local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                            idx=idx)
                        w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
                        party_models_state.append(copy.deepcopy(w))
                        party_losses.append(loss)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("| WARNING: ran out of memory, trying to clear cache")
                            torch.cuda.empty_cache()
                            local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality,
                                                idx=idx)
                            w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
                            party_models_state.append(copy.deepcopy(w))
                            party_losses.append(loss)
                        else:
                            raise e

            if fusion_key == 'FedAvg':
                current_model_state_dict = FedAvg(party_models_state)
                model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
                party_models_dict[fusion_key] = party_models_state
            else:
                current_model_state_dict = FedAvg(party_models_state[1:])
                model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
                party_models_dict[fusion_key] = party_models_state

            # testing
            if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
                eval_glob = DenseNetModel(args).to(args.device)
            elif args.dataset == 'OCT':
                eval_glob = VGG16Model().to(args.device)
            eval_glob.load_state_dict(model_dict[fusion_key])
            eval_glob.eval()

            clean_acc, loss_test = test_img(eval_glob, clean_dataset_test, args)
            clean_accuracy[fusion_key][iter] = clean_acc
            print(f'Global Clean Accuracy {fusion_key}, round {iter} = {clean_acc}')
            pois_acc, loss_poisoned_test = test_img(eval_glob, poisoned_dataset_test, args)
            pois_accuracy[fusion_key][iter] = pois_acc
            print(f'Global Backdoor Accuracy {fusion_key}, round {iter} = {pois_acc}')
    print('------------------------------------------------------')
    clip_grad = 5

    if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
        initial_model = DenseNetModel(args).to(args.device)
    elif args.dataset == 'OCT':
        initial_model = VGG16Model().to(args.device)
    unlearned_model_dict = {}

    for fusion_key in fusion_types_unlearn:
        if fusion_key == 'Retrain':
            unlearned_model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

    clean_accuracy_unlearn = {}
    pois_accuracy_unlearn = {}

    for fusion_key in fusion_types_unlearn:
        clean_accuracy_unlearn[fusion_key] = 0
        pois_accuracy_unlearn[fusion_key] = 0

    for fusion_key in fusion_types:
        if fusion_key == 'Retrain':
            continue

        if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
            initial_model = DenseNetModel(args).to(args.device)
        elif args.dataset == 'OCT':
            initial_model = VGG16Model().to(args.device)
        fedavg_model_state_dict = copy.deepcopy(model_dict[fusion_key])
        fedavg_model = copy.deepcopy(initial_model)
        fedavg_model.load_state_dict(fedavg_model_state_dict)

        party_models = copy.deepcopy(party_models_dict[fusion_key])
        party0_model = copy.deepcopy(party_models_dict[fusion_key][0])


        # Load the state_dict into a new model
        temp_model = copy.deepcopy(initial_model)  # create a new model instance
        temp_model.load_state_dict(party0_model)  # load the state_dict

        model_ref_vec = args.num_users / (args.num_users - 1) * torch.nn.utils.parameters_to_vector(
            fedavg_model.parameters()) \
                        - 1 / (args.num_users - 1) * torch.nn.utils.parameters_to_vector(temp_model.parameters())

        # compute threshold
        model_ref = copy.deepcopy(initial_model)
        torch.nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())
        eval_model = copy.deepcopy(model_ref)
        unlearn_clean_acc, unlearn_loss_test = test_img(eval_model, clean_dataset_test, args)
        print(f'Clean Accuracy for Reference Model = {unlearn_clean_acc}')
        unlearn_pois_acc, unlearn_pois_acc = test_img(eval_model, poisoned_dataset_test, args)
        print(f'Backdoor Accuracy for Reference Model = {unlearn_pois_acc}')

        dist_ref_random_lst = []
        for _ in range(10):
            if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
                dist_ref_random_lst.append(Distance.get_distance(model_ref, DenseNetModel(args).to(args.device)))
            elif args.dataset == 'OCT':
                dist_ref_random_lst.append(Distance.get_distance(model_ref, VGG16Model().to(args.device)))

            dist_ref_random_lst = [tensor.cpu() for tensor in dist_ref_random_lst]
        print(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
        threshold = np.mean(dist_ref_random_lst) / 3
        print(f'Radius for model_ref: {threshold}')
        dist_ref_party = Distance.get_distance(model_ref, temp_model)
        print(f'Distance of Reference Model to party0_model: {dist_ref_party}')

    print("----------------------unlearning--------------------------------")

    model = copy.deepcopy(model_ref)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), args.lr, args.momentum)
    model.train()
    flag = False

    for epoch in range(args.num_local_epochs_unlearn):
        print('------------', epoch)
        if flag:
            break
        for batch_id, (x_batch, y_batch) in enumerate(trainloader_lst[0]):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device).long()
            opt.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_joint = -loss  # negate the loss for gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            with torch.no_grad():
                distance = Distance.get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = torch.nn.utils.parameters_to_vector(
                        model.parameters()) - torch.nn.utils.parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * np.sqrt(threshold)
                    proj_vec = torch.nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    torch.nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    distance = Distance.get_distance(model, model_ref)

            distance_ref_party_0 = Distance.get_distance(model, temp_model)
            print('Distance from the unlearned model to party 0:', distance_ref_party_0.item())

            if distance_ref_party_0 > args.distance_threshold:
                flag = True
                break

    unlearned_model = copy.deepcopy(model)
    unlearned_model_dict[fusion_types_unlearn[1]] = unlearned_model.state_dict()

    if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
        eval_model = DenseNetModel(args).to(args.device)
    elif args.dataset == 'OCT':
        eval_model = VGG16Model().to(args.device)
    eval_model.load_state_dict(unlearned_model_dict[fusion_types_unlearn[1]])
    unlearn_clean_acc, unlearn_loss_test = test_img(eval_model, clean_dataset_test, args)
    print(f'Clean Accuracy for UN-Local Model = {unlearn_clean_acc}')
    clean_accuracy_unlearn[fusion_types_unlearn[1]] = unlearn_clean_acc
    pois_unlearn_acc, pois_unlearn_loss = test_img(eval_model, poisoned_dataset_test, args)
    print(f'Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}')
    pois_accuracy_unlearn[fusion_types_unlearn[1]] = pois_unlearn_acc

    print("----------------------Post-Training--------------------------------")

    num_fl_after_unlearn_rounds = args.epochs

    clean_accuracy_unlearn_fl_after_unlearn = {}
    pois_accuracy_unlearn_fl_after_unlearn = {}
    loss_unlearn = {}

    for fusion_key in fusion_types_unlearn:
        clean_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
        pois_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
        loss_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)

    for round_num in range(num_fl_after_unlearn_rounds):

        for fusion_key in fusion_types_unlearn:
            # Reduce num_parties by 1 to remove the erased party
            current_model_state_dict = copy.deepcopy(unlearned_model_dict[fusion_key])
            if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
                current_model = DenseNetModel(args).to(args.device)
            elif args.dataset == 'OCT':
                current_model = VGG16Model().to(args.device)
            current_model.load_state_dict(current_model_state_dict)

            party_models_state = []
            party_losses = []

            print("-----------------local training round--------------")

            idxs_users = [i for i in range(args.num_users)]
            print(idxs_users)
            for idx in idxs_users:
                local_dataset = train_datasets[idx]
                local = LocalUpdate(args=args, dataset=local_dataset, loss_global=loss_avg, quality=quality, idx=idx)
                w, loss, loss_diff = local.train(net=copy.deepcopy(current_model).to(args.device))
                party_models_state.append(copy.deepcopy(w))
                party_losses.append(loss)

            current_model_state_dict = FedAvg(party_models_state)
            model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
            party_models_dict[fusion_key] = party_models_state
            if args.dataset == 'covid19' or args.dataset == 'brain' or args.dataset == 'covid':
                eval_model = DenseNetModel(args).to(args.device)
            elif args.dataset == 'OCT':
                eval_model = VGG16Model().to(args.device)

            eval_model.load_state_dict(current_model_state_dict)
            unlearn_clean_acc, unlearn_loss_test = test_img(eval_model, clean_dataset_test, args)
            print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {unlearn_clean_acc}')
            clean_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_clean_acc
            unlearn_pois_acc, unlearn_pois_loss = test_img(eval_model, poisoned_dataset_test, args)
            print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {unlearn_pois_acc}')
            pois_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_pois_acc
