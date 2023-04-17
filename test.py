#coding=utf-8
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# dataset
# Set up according to your situation
set = 'CUB'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # model path
    pth_path = "./models/cub_epoch144.pth"
    num_classes = 200
elif set == 'Aircraft':
    root = './datasets/FGVC-aircraft'  # dataset path
    # model path
    pth_path = "./models/air_epoch146.pth"
    num_classes = 100

batch_size = 10

#load dataset
_, testloader = read_dataset(input_size, batch_size, root, set)

#  Set up num_classes according to your situation
num_classes = ''
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

#checkpoint
pth_path = './'
epoch = auto_load_resume(model, pth_path, status='test')


print('Testing')
raw_correct = 0
local_correct = 0
sum_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(testloader)):
        if set == 'CUB':
            x, y, boxes, _ = data
        else:
            x, y = data
        x = x.to(DEVICE)
        labels = y.to(DEVICE)

        raw_logits, local_logits = model(x, epoch, i, 'train')
        pred = raw_logits.max(1, keepdim=True)[1]
        raw_correct += pred.eq(labels.view_as(pred)).sum().item()

        pred = local_logits.max(1, keepdim=True)[1]
        local_correct += pred.eq(labels.view_as(pred)).sum().item()

        sum_logit = raw_logits + local_logits
        pred = sum_logit.max(1, keepdim=True)[1]
        sum_correct += pred.eq(labels.view_as(pred)).sum().item()

    # output in yourself