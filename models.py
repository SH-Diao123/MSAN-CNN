#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:15:08 2020

@author: dsh
"""

import torchvision.models as models
import torch
import torch.nn as nn
import os


class mul_scale_attn(nn.Module):
    def __init__(self, num_classes=2):
        super(mul_scale_attn, self).__init__()
        self.num_classes = num_classes
        modelBackbone1 = list(models.vgg19_bn(pretrained=True).features.children())
        modelBackbone2 = list(models.vgg19_bn(pretrained=True).features.children())
        
        # self.backbone1 = models.vgg19_bn(pretrained=True, num_classes=self.num_classes).features
        # self.backbone2 = models.vgg19_bn(pretrained=True, num_classes=self.num_classes).features
        self.backbone1 = nn.Sequential(*modelBackbone1)
        self.backbone2 = nn.Sequential(*modelBackbone2)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.conv = nn.Conv2d(512, 1, (7, 7))
        self.sigmoid = nn.Sigmoid()
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=self.num_classes, bias=True),
            )
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=self.num_classes, bias=True),
            )
        
        
        
        
    def forward(self, x1, x2):
        # print(self.backbone1)
        # assert (x1.shape[2]*2, x1.shape[3]*2) == (x2.shape[2], x2.shape[3])
        #get feature
        self.feature1 = self.pool(self.backbone1(x1))
        self.feature2 = self.pool(self.backbone2(x2))
        ###get atttention
        # |cosx|
        self.attention = torch.abs(torch.cos(self.conv(self.feature1)))
        # sigmoid
        # self.attention = self.sigmoid(self.conv(self.feature1))
        
        self.attention = torch.squeeze(torch.squeeze(self.attention, 3), 2)
        #classifier
        self.feature1 = torch.flatten(self.feature1, 1)
        self.feature2 = torch.flatten(self.feature2, 1)
        self.output1 = self.attention * self.classifier1(self.feature1)
        self.output2 = (1 - self.attention) * self.classifier2(self.feature2)
        return (self.output1 + self.output2)
        # return (self.classifier1(self.feature1) + self.classifier2(self.feature2))
        


