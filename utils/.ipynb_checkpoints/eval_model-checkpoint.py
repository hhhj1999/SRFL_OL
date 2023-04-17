import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, set, vis_num
from utils.cal_iou import calculate_iou
from utils.vis import image_with_boxes

def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct = 0
    local_correct = 0
    sum_correct = 0

#     ppp = 1
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            if set == 'CUB':
                images, labels, boxes, scale = data
            else:
                images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            raw_logits,local_logits = model(images, epoch, i,labels, status)
#             raw_logits,_ = model(images)
            

            raw_loss = criterion(raw_logits, labels)


            total_loss = raw_loss

            raw_loss_sum += raw_loss.item()

            total_loss_sum += total_loss.item()

            # correct num
            # raw
            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()

            # raw
            pred = local_logits.max(1, keepdim=True)[1]
            local_correct += pred.eq(labels.view_as(pred)).sum().item()


            sum_logit = raw_logits + local_logits
            pred = sum_logit.max(1, keepdim=True)[1]
            sum_correct += pred.eq(labels.view_as(pred)).sum().item()



    # raw_loss_avg = raw_loss_sum / (i+1)
    # local_loss_avg = local_loss_sum / (i+1)
    # windowscls_loss_avg = windowscls_loss_sum / (i+1)
    # total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    local_accuracy = local_correct / len(testloader.dataset)
    sum_accuracy = sum_correct / len(testloader.dataset)


    return raw_accuracy,local_accuracy,sum_accuracy