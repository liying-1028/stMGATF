# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:39:43 2022

@author: dell
"""
import os
import glob
import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from datetime import datetime
import pandas as pd
#random.seed ( 123)
from tensorboardX import SummaryWriter
from torch.autograd import Variable
args={
    'no_cuda':True,
    #'no_cuda':False,
    'fastmode':False,
    'seed':2,
    'epochs':10000,
    'lr':0.0005,
    'weight_decay':0,
    'hidden':[64,8],
    'nb_heads':[1,8],
    'dropout':0,
    'alpha':0.05,
    'patience':110,
    'batch_size':24
}

args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
#F:/DESKTOP/k_mea
#path1 = 'F:/DESKTOP/k_mea/h1_151673_0.2436.csv'
path1 = 'F:/DESKTOP/DLPFC/151673/h1_151673_0.2371_8.csv'
A1 = pd.read_csv(path1, header=0, index_col=0)
#path2 = 'F:/DESKTOP/DLPFC/151673/h2_151673_0.6014_8.csv'
path2 = 'F:/DESKTOP/DLPFC/151673/h2_151673_0.5814.csv'
B1 = pd.read_csv(path2, header=0, index_col=0)
path3 = 'F:/DESKTOP/DLPFC/151673/h3_151673_0.2126_8.csv'
C1 = pd.read_csv(path3, header=0, index_col=0)
h1 = torch.from_numpy(np.float32(A1.values))
h2 = torch.from_numpy(np.float32(B1.values))
h3 = torch.from_numpy(np.float32(C1.values))
inputs = torch.stack([h1, h2, h3], dim=0)


class_file = 'Annotation_train_test_split.csv'
class_data = pd.read_csv(class_file, header=0, index_col=0)
labels = encode_onehot(class_data.Cluster.array)
labels = torch.LongTensor(np.where(labels)[1])
#labels = torch.Tensor(np.where(labels)[1]).long()


idx_train = range(3000)
idx_val = range(3000, 3500)
idx_test = range(3500, 3639)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


import torch
from torch import nn

import torch.nn as nn
import torch


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=3,kernel_size=7):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, int(in_channels / rate) ,kernel_size, padding=3),#in_channels*in_channels/rate   conv2d改为conv1d
            nn.BatchNorm1d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(in_channels / rate), out_channels,kernel_size, padding=3),##in_channels/rate*out_channels
            nn.BatchNorm1d(out_channels)
        )

    def inference(self, x):
        c, h, w = x.shape
        x_permute = x.permute(1, 2, 0).view(-1, c)
        x_att_permute = self.channel_attention(x_permute).view(h, w, c)
        x_channel_att = x_att_permute.permute(2, 0, 1)

        x = x * x_channel_att
    ######修改
        x=x.permute(2, 0, 1)#10*3*3639

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att #10*3#3639
        out = out.permute(2, 0, 1)
        out = out.permute(2,0,1)
        out=out[0]+out[1]+out[2]

        return out,F.log_softmax(out, dim=1)

    def forward(self, x):
        h_robust, class_prediction = self.inference(inputs)
        return h_robust, class_prediction

'''
if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 48)
    b, c, h, w = x.shape
    net = GAM_Attention(in_channels=c, out_channels=c)
    y = net(x)
'''
c, h, w = inputs.shape
model  = GAM_Attention(in_channels=c, out_channels=c)
print(model)

optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])

if args['cuda']:
    model.cuda()
    labels=labels.cuda()
labels=Variable(labels)

def train(epoch,writer):
    t = time.time()
    model.train()
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    h_robust ,class_prediction= model.inference(inputs)

    #mu_f = fusion
    #outputs_f=F.softmax(fusion,dim=1)
    #output已经做了log_softmax所以不需要CrossEntropyLoss
    class_p = class_prediction.max(1)[1].type_as(labels)
    loss_train = F.nll_loss(class_prediction[idx_train], labels[idx_train])
    acc_train = accuracy(class_prediction[idx_train], labels[idx_train])
    #writer.add_scalar('loss_train',loss_train,epoch)
    #writer.add_scalar('acc_train',acc_train,epoch)
    loss_train.backward()
    optimizer.step()

    if not args['fastmode']:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
       # mu_f = fusion
      #  outputs_f = F.softmax(fusion,dim=1)
    h_robust, class_prediction = model.inference(inputs)

    loss_val = F.nll_loss(class_prediction[idx_val], labels[idx_val])
    acc_val = accuracy(class_prediction[idx_val], labels[idx_val])
    writer.add_scalar('loss_val',loss_val,epoch)
    writer.add_scalar('acc_val',acc_val,epoch)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    #outputs_f = F.softmax(fusion,dim=1)
    h_robust, class_prediction = model.inference(inputs)
    preds_m = class_prediction.max(1)[1].type_as(labels)
    preds_m = preds_m.numpy()
    kmeans = KMeans(n_clusters=7)
    preds_k = kmeans.fit_predict(h_robust.data.cpu().numpy())
    cc = metrics.adjusted_rand_score(labels[idx_test], preds_m[idx_test])#ARI
    cc1 = normalized_mutual_info_score(labels[idx_test], preds_m[idx_test])#NMI
    kk = metrics.adjusted_rand_score(labels, preds_k)  # ARI
    kk1 = normalized_mutual_info_score(labels, preds_k)  # NMI
    loss_test = F.nll_loss(class_prediction[idx_test], labels[idx_test])
    acc_test = accuracy(class_prediction[idx_test], labels[idx_test])
    he = h_robust.detach().numpy()
    SC = silhouette_score(he, preds_k)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    print('models_h''ARI:', cc, 'NMI:', cc1)
    print('kmeans','ARI:',kk,'NMI:',kk1)
    print('ASW:',SC)
    mu_f=h_robust.detach().numpy()
    np.savetxt( "he_151673_0.6253.csv", mu_f, delimiter="," )
    np.savetxt("pred_151673_0.6253.csv", preds_k, delimiter=",")
    return mu_f









def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



t_total = time.time()
loss_values = []
bad_counter = 0
best = args['epochs'] + 1
best_epoch = 0
writer=SummaryWriter('./log/'+datetime.now().strftime('%Y%m%d-%H%M%S'))

for epoch in range(args['epochs']):
    loss_values.append(train(epoch,writer))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args['patience']:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()

torch.cuda.empty_cache()

mu_robust, class_prediction = model.inference(inputs)
fusion=mu_robust
datarobust = mu_robust.detach().numpy()
dataclass_prediction = class_prediction.detach().numpy()
