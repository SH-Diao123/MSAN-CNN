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
import matplotlib.pyplot as plt
from models import mul_scale_attn
from sklearn.metrics import roc_curve, auc




torch.manual_seed(1314)
torch.cuda.manual_seed(1314)


def dataloader_diao():
    a_path = ''
    b_path = ''
    
    
    a_file = os.listdir(a_path)
    b_file = os.listdir(b_path)

    all_train_file_name, all_train_file_label = [], []
    all_test_file_name, all_test_file_label = [], []

    for i in range(0, len(a_file)):
        all_train_file_name.append(a_path+a_file[i])
        all_train_file_label.append(0)
    for i in range(0, len(b_file)):
        all_train_file_name.append(b_path+b_file[i])
        all_train_file_label.append(1)

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

    return random.shuffle(all_train_file_name), random.shuffle(all_train_file_label), \
           all_test_file_name,all_test_file_label


def get_kfold_data(k, i, X, y):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = X[0:val_start] + X[val_end:]
        y_train = y[0:val_start] + y[val_end:]
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


def dataloader_k_fold(k, index, is_tese=False):
    
    train_path = [
        '',
        ''
    ]
    # 16x-32x  good_count is 6785, bad_count is 6672
    test_path = [
        '',
        ''
    ]

    num_classes = len(train_path)
    classes_label = [i for i in range(num_classes)]
    train_files, test_files = [], []
    # train_files = [
    #                   [tr_file_list_0, tr_label_list_0]   # classes:0
    #                   [tr_file_list_1, tr_label_list_1]   # classes:1
    #                   ...
    #               ]
    for i in range(num_classes):
        tr_files = os.listdir(train_path[i])
        te_files = os.listdir(test_path[i])
        tr_file_list, tr_label_list = [], []
        te_file_list, te_label_list = [], []
        for j in range(len(tr_files)):
            tr_file_list.append(train_path[i] + tr_files[j])
            tr_label_list.append(classes_label[i])
        for j in range(len(te_files)):
            te_file_list.append(test_path[i] + te_files[j])
            te_label_list.append(classes_label[i])
        train_files.append([tr_file_list, tr_label_list])
        test_files.append([te_file_list, te_label_list])


    all_train_x, all_train_y = [], []
    all_valid_x, all_valid_y = [], []
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    for i in range(num_classes):
        fold_size = len(train_files[i][0]) // k  # 每份的个数:数据总条数/折数（组数）
        val_start = index * fold_size
        if index != k - 1:
            val_end = (index + 1) * fold_size
            X_valid, y_valid = train_files[i][0][val_start : val_end], train_files[i][1][val_start : val_end]
            X_train = train_files[i][0][0 : val_start] + train_files[i][0][val_end : ]
            y_train = train_files[i][1][0 : val_start] + train_files[i][1][val_end : ]
        else:  # 若是最后一折交叉验证
            X_valid, y_valid = train_files[i][0][val_start : ], train_files[i][1][val_start : ]  # 若不能整除，将多的case放在最后一折里
            X_train = train_files[i][0][0 : val_start]
            y_train = train_files[i][1][0 : val_start]

        all_train_x += X_train
        all_train_y += y_train
        all_valid_x += X_valid
        all_valid_y += y_valid


    if is_tese:
        test_x, test_y = [], []
        for i in range(num_classes):
            test_x += test_files[i][0]
            test_y += test_files[i][1]
        return test_x, test_y

    return all_train_x, all_train_y, all_valid_x, all_valid_y

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
             i = ((outputs[k] * labels[k])>0).sum()
             u = ((outputs[k] + labels[k])>0).sum()
             scores[k] = i.float() / u.float() if u != 0 else u
         return scores.mean()
    

def train(k, X_train, y_train, X_val, y_val, BATCH_SIZE, lr, epochs):
    all_train_file_name,all_train_file_label,all_test_file_name,all_test_file_label = dataloader_diao()
    '''
    ###baseline dataloader
    TrainImgLoader = torch.utils.data.DataLoader(
             myImageFloder(all_train_file_name,all_train_file_label, True), 
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=True)
    
    testImgLoader = torch.utils.data.DataLoader(
             myImageFloder(all_test_file_name,all_test_file_label, False), 
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=False)
    '''
    ### attention dataloader
    TrainImgLoader = torch.utils.data.DataLoader(
             myImageFloderAttn(X_train, y_train, True),
             batch_size= BATCH_SIZE, shuffle= True, num_workers= 4, drop_last=True)

    testImgLoader = torch.utils.data.DataLoader(
             myImageFloderAttn(X_val, y_val, False),
             batch_size= BATCH_SIZE * 4, shuffle= True, num_workers= 4, drop_last=False)

    # models = models.vgg19_bn(num_classes=2)
    models = mul_scale_attn(num_classes=2).cuda()

    # Loss and Optimizer
    # www = torch.tensor([1.0, 1.12])    
    criterion = nn.CrossEntropyLoss(weight=www).cuda()
    optimizer = torch.optim.Adam(models.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    PATH = ''%k

    # # Training
    max_losses = 1e10
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        # for i, (imagesL, imagesH, labels) in enumerate(TrainImgLoader):
        #     # images = Variable(images.cuda())
        #     imagesL, imagesH = Variable(imagesL.cuda()), Variable(imagesH.cuda())
        #     labels = Variable(labels.cuda())
        #     models.train()
        #     # Forward + Backward + Optimize
        #     optimizer.zero_grad()
        #     outputs = models(imagesL, imagesH)
        #     loss = criterion(outputs, labels)
        #     acc = accuracy(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     train_losses.append(loss.item())
        #     train_acc.append(acc)
        #
        #     if (i + 1) % (int(len(TrainImgLoader)/6)) == 0:
        #         print("Epoch [%d/%d], Iter [%d/%d],  Loss= %.4f, ACC is %.4f" %
        #               (epoch + 1, epochs, i + 1, len(TrainImgLoader), loss.item(), acc))


        with torch.no_grad():
            models.load_state_dict(torch.load(PATH))
            models.eval().cuda()
            val_loss = 0
            correct = 0
            tar_tmp, per_tmp = [], []
            for i, (imagesL, imagesH, labels) in enumerate(testImgLoader):
                # test_image = Variable(images.cuda())
                test_imageL, test_imageH = Variable(imagesL.cuda()), Variable(imagesH.cuda())
                the_labels = Variable(labels.cuda())
                outputs = models(test_imageL, test_imageH)
                val_loss += criterion(outputs, the_labels)
                correct += accuracy(outputs, the_labels)

                _, predicted = torch.max(outputs.data, 1)
                tar_tmp += the_labels.view(-1).tolist()
                per_tmp += predicted.view(-1).tolist()

            if ((epoch + 1) == epochs):
                con_x = metrics.confusion_matrix(tar_tmp, per_tmp)
                print('final epoch:', con_x)
                fprt, tprt, _ = roc_curve(tar_tmp, per_tmp, pos_label=1)
                auc_valuet = auc(fprt, tprt)
                acc_sorce = metrics.accuracy_score(tar_tmp, per_tmp)
                print('final epoch auc_value is %2f '%auc_valuet)
                print('test_acc is ', acc_sorce)
                print('test_auc is ', auc_valuet)
                print('test_spe is ', fprt[1])
                print('test_sen is ', tprt[1])

        #     val_loss /= i + 1
        #     correct /= i+1
        #     val_losses.append(val_loss.item())
        #     val_acc.append(correct)
        #     print('Epoch [%d/%d], Val_loss= %.4f, ACC is %.4f'%(epoch + 1, epochs, val_loss.item(), correct))
        #
        #     if val_loss < max_losses:
        #         max_losses = val_loss
        #         print('----------Now has more lower loss in No. %d epoch ----------'%(epoch+1))
        #         torch.save(models.state_dict(), PATH)
        #
        # # Decaying Learning Rate
        # scheduler.step()

    return acc_sorce, auc_valuet, fprt[1], tprt[1]
    # return train_losses, val_losses, train_acc, val_acc


def k_fold_train(bp=0, k=5, num_epochs=30, learning_rate=1e-3, batch_size=16):
    train_loss_sum, valid_loss_sum = [], []
    train_acc_sum, valid_acc_sum = [], []
    # all_train_file_name, all_train_file_label, all_test_file_name, all_test_file_label = dataloader_k_fold()

    for i in range(bp, k):
        print('*' * 25, '  Training  第 ', i + 1, '折', '*' * 25)
        # 每份数据进行训练
        xt, yt, xv, yv = dataloader_k_fold(k, i)
        train_loss, val_loss, train_acc, val_acc = train(i, xt, yt, xv, yv, batch_size, learning_rate, num_epochs)

        # print('train_loss:{:.4f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
        # print('valid loss:{:.4f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))

        # train_loss_sum.append(train_loss[-1])
        # valid_loss_sum.append(val_loss[-1])
        # train_acc_sum.append(train_acc[-1])
        # valid_acc_sum.append(val_acc[-1])

        train_loss_sum.append(train_loss)
        valid_loss_sum.append(val_loss)
        train_acc_sum.append(train_acc)
        valid_acc_sum.append(val_acc)


    train_loss_sum, valid_loss_sum = np.array(train_loss_sum), np.array(valid_loss_sum)
    train_acc_sum, valid_acc_sum = np.array(train_acc_sum), np.array(valid_acc_sum)
    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)
    # print('\n', 'every train loss is ', train_loss_sum)
    # print('\n', 'every train acc is ', train_acc_sum)
    #
    # print('\n', '#' * 30)
    # print('\n', 'every validation loss is ', valid_loss_sum)
    # print('\n', 'every validation acc is ', valid_acc_sum)
    #
    # print('\n', '#' * 30)
    print('\n', 'average train loss:{:.4f}, average train accuracy:{:.3f}%'.
          format(np.mean(train_loss_sum), np.mean(train_acc_sum)))
    print('\n', 'average train loss:{:.4f}, average train accuracy:{:.3f}%'.
          format(np.std(train_loss_sum), np.std(train_acc_sum)))
    print('\n', 'average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.
          format(np.mean(valid_loss_sum), np.mean(valid_acc_sum)))
    print('\n', 'average train loss:{:.4f}, average train accuracy:{:.3f}%'.
          format(np.std(valid_loss_sum), np.std(valid_acc_sum)))




if __name__ == '__main__':
    k_fold_train(bp=1, k=5, num_epochs=1, learning_rate=1e-3, batch_size=16)
