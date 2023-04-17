import torch
import os
from datasets import dataset, dataset_NA


def read_dataset(input_size, batch_size, root, set):
    if set == 'CUB':
        print('Loading CUB trainset')
        trainset = dataset.CUB(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading CUB testset')
        testset = dataset.CUB(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'CAR':
        print('Loading car trainset')
        trainset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading car testset')
        testset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Aircraft':
        print('Loading Aircraft trainset')
        trainset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading Aircraft testset')
        testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Dog':
        print('Loading Dog trainset')
        trainset = dataset.dogs(root=root, train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading Dog testset')
        testset = dataset.dogs(root=root, train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'NA':
        print('Loading NA trainset')
        # trainset = dataset.dogs(root=root, train=True)
        root_NA_train= '/public/home/jd_wgc/zsj_boom/2023-upupup/NABirds/train/'
        trainset = dataset_NA.ImageDataset(istrain=True,root=root_NA_train,data_size=448)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=1, drop_last=False)
        print('Loading NA testset')
        root_NA_test = '/public/home/jd_wgc/zsj_boom/2023-upupup/NABirds/test/'
        testset = dataset_NA.ImageDataset(istrain=False,root=root_NA_test,data_size=448)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=1, drop_last=False)
    else:
        print('Please choose supported dataset')
        os._exit()

    return trainloader, testloader