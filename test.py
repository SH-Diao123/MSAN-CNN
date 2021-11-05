from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
import math
from data_loader import myImageFloder, myImageFloderAttn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from models import mul_scale_attn
from scipy.io import savemat

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.manual_seed(1314)
torch.cuda.manual_seed(1314)


def dataloader_diao():
    all_test_file_name, all_test_file_label = [], []

    aa_path = ''
    bb_path = ''
    aa_file = os.listdir(aa_path)
    bb_file = os.listdir(bb_path)

    for i in range(0, len(aa_file)):
        all_test_file_name.append(aa_path+aa_file[i])
        all_test_file_label.append(0)
    for i in range(0, len(bb_file)):
        all_test_file_name.append(bb_path+bb_file[i])
        all_test_file_label.append(1)

    return all_test_file_name,all_test_file_label


def accuracy(output, target):
    correct = 0
    total = 0
    with torch.no_grad():
        _, pred = torch.max(output.data, 1)
        correct += torch.sum(pred == target)
        total += len(target)
        return 100.0 * float(correct) / float(total)


def IoU(outputs, labels):
    with torch.no_grad():
         scores = np.zeros(len(outputs))
         #_, outputs = torch.max(outputs.data, 1)
         #_, labels = torch.max(labels.data, 1)
         for k in range(len(outputs)):
             u = ((outputs[k] + labels[k])>0).sum()
             scores[k] = i.float() / u.float() if u != 0 else u
         return scores.mean()
    


BATCH_SIZE = 16

all_test_file_name,all_test_file_label = dataloader_diao()

testImgLoader = torch.utils.data.DataLoader(
         myImageFloderAttn(all_test_file_name,all_test_file_label, False), 
         batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=False)


resnet = mul_scale_attn(num_classes=2).cuda()

# Loss and Optimizer
www = torch.tensor([1.0, 1.12])
criterion = nn.CrossEntropyLoss(weight=www).cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
PATH = ''

resnet.load_state_dict(torch.load(PATH))
resnet.eval().cuda()
with torch.no_grad():
     pre, tar = [], []
     loss=0.0
     for i, (imagesL, imagesH, labels) in enumerate(testImgLoader):
         imagesL, imagesH = Variable(imagesL.cuda()), Variable(imagesH.cuda())
         labels = Variable(labels.cuda())
         optimizer.zero_grad()
         outputs = resnet(imagesL, imagesH)
         loss += criterion(outputs, labels)
         _, pred = torch.max(outputs.data, 1)
         pre += pred.view(-1).tolist()
         tar += labels.view(-1).tolist()
     loss /= i + 1
     mar = metrics.confusion_matrix(tar, pre)
     
     # save data to mat
     # mat_file = PATH.split('.')[0] + '.mat'
     # savemat(mat_file, {'predict':pre, 'ground_truth':tar})
     
     
     print('val_loss= %.4f'%(loss.item()))
     print(mar)
     acc_sorce = metrics.accuracy_score(tar, pre)
     print('Acc is ', acc_sorce)


     #compute auc and show
     fpr, tpr, thre = roc_curve(tar, pre, pos_label=1)
     auc_value = auc(fpr, tpr)
     print('thertholds is ', thre)
     print('auc_value is %2f'%auc_value)
     print('tpr is ', tpr)
     print('fpr is ', fpr)
     plt.figure()
     plt.plot(fpr, tpr, color='darkorange', lw=2, label='Mul_Scale_Attn_16x(AUC={:4f})'.format(auc_value))
     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
     plt.xlim([-0.05, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('ROC')
     plt.legend(loc='lower right')
     plt.show()

