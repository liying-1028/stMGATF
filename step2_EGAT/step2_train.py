from __future__ import division
from __future__ import print_function

import os
import glob
import time
from datetime import datetime
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import load_data, accuracy,DSN
from models import GAT
from sklearn import metrics

# Training settings
args={
    'no_cuda':True,
    #'no_cuda':False,
    'fastmode':False,
    'seed':1,#调参至22
    'epochs':10000,
    'lr':0.0005,
    'weight_decay':0,
    'hidden':[64,8],
    'nb_heads':[1,8],
    'dropout':0,
    'alpha':0.005,
    'patience':110,
    'batch_size':24
}
#lr0.005 alpha 0.05
args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])

#%%Load data
edge_attr, features, labels, idx_train, idx_val, idx_test = load_data()
print('Loading data Successfully')
#%%reshuffle data
#需要注意的adj和features, labels的Node顺序得一致。对adj乱序得分别从两个维度打乱。
"""
index = [i for i in range(adj.shape[0])]
random.shuffle(index)
tem_adj=torch.empty_like(adj)
temp_adj=torch.empty_like(adj)
tem_features=torch.empty_like(features)
tem_labels=torch.empty_like(labels)
for i in range(len(index)):
    tem_adj[i]=adj[index[i]]
    tem_features[i]=features[index[i]]
    tem_labels[i]=labels[index[i]]

for i in range(len(index)):
    temp_adj[:,i]=tem_adj[:,index[i]]

adj=temp_adj
features=tem_features
labels=tem_labels
"""

#using batch training
#batch是对于multigraph来说的，训练的时候还是得把所有的节点一次输进去，因为信息需要从邻居传递


#%%Model and optimizer

def loss_function(preds, labels, mu, logvar, n_nodes, norm,
                  lamda=None, robust_rep=None, prediction=None,
                  true_class=None,  loc=0):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    if robust_rep is not None and lamda is not None:
        lamda = lamda.detach()
        robust_rep = robust_rep.detach()

        robust_loss = torch.sum(lamda[:, loc] * (torch.sum((mu - robust_rep) ** 2, dim=1)))
    # print(robust_loss)

    else:
        robust_loss = torch.tensor(0.0)

    if logvar is None:
        KLD = torch.tensor(0.0)
    else:
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    if prediction is None:
        predict_class = torch.tensor(0.0)

    else:
        predict_class_sum = F.cross_entropy(prediction, true_class)
        predict_class = torch.mean(predict_class_sum)

    return cost, KLD, robust_loss, predict_class


model = GAT(nfeat=features.shape[1],
            ef_sz=tuple(edge_attr.shape),
            nhid=args['hidden'], 
            nclass=int(labels.max()) + 1,
            dropout=args['dropout'], 
            nheads=args['nb_heads'], 
            alpha=args['alpha'])


#model.load_state_dict(torch.load('F:/Lu/151673/h1_151673/EGAT/4376.pkl'))

optimizer = optim.Adam(model.parameters(),
                       lr=args['lr'], 
                       weight_decay=args['weight_decay'])

if args['cuda']:
    model.cuda()
    features = features.cuda()
    edge_attr = edge_attr.cuda()
    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

features, edge_attr, labels = Variable(features), Variable(edge_attr), Variable(labels)


def train(epoch,writer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    #output已经做了log_softmax所以不需要CrossEntropyLoss
    recovered,mu,output = model.inference(features, edge_attr)
    n_spot=3639
    adj_label = np.eye(3639)
    adj_label=adj_label.tolist()
    adj_label=np.array(adj_label)
    adj_label= torch.tensor(adj_label)  # 将array a 转换为tensor

    cost, KLD, robust_loss, pre_cost = loss_function(preds=recovered, labels=adj_label,
                                                     mu=mu, logvar=None,
                                                     n_nodes=torch.as_tensor(n_spot).cuda(),
                                                     norm=0.5008,
                                                     lamda=None, robust_rep=None,
                                                     prediction=output, true_class=labels, loc=0)#尽量把recovered改成3000×3000的

    loss_train = cost + KLD +  0.0005 * robust_loss + 8 * pre_cost

    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    writer.add_scalar('loss_train',loss_train,epoch)
    writer.add_scalar('acc_train',acc_train,epoch)
    loss_train.backward()
    optimizer.step()

    if not args['fastmode']:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        recovered,mu, output = model.inference(features, edge_attr)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
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
    recovered, mu,output = model.inference(features, edge_attr)
    preds_m = output.max(1)[1].type_as(labels)
    preds_m = preds_m.numpy()
    kmeans = KMeans(n_clusters=int(labels.max()))
    preds_k = kmeans.fit_predict(mu.data.cpu().numpy())
    cc = metrics.adjusted_rand_score(labels[idx_test], preds_m[idx_test])#ARI
    cc1 = normalized_mutual_info_score(labels[idx_test], preds_m[idx_test])#NMI
    kk = metrics.adjusted_rand_score(labels, preds_k)  # ARI
    kk1 = normalized_mutual_info_score(labels, preds_k)  # NMI
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    print('models''ARI:', cc, 'NMI:', cc1)
    print('kmeans','ARI:',kk,'NMI:',kk1)
    mu=mu.detach().numpy()
    np.savetxt( "h1_DLPFC_.csv", mu, delimiter="," )
    np.savetxt("pred_m.csv", preds_m, delimiter=",")

#%%Train model
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
