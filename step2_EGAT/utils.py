import numpy as np
import scipy.sparse as sp
import torch
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.spatial import distance


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sigmoid_function(z):
    return np.exp(-z)



def load_data(path="./stMGATF_test_data/DLPFC_151673/"):
    """Load citation network dataset (cora only for now)"""
    RNA_file = path + 'stMGATF/128_8e-05-2000-1000-50-1000-2000_RNA_AE_latent.csv'
    imageRep_file = path + 'stMGATF/128_0.5_200_128_simCLR_reprensentation.csv'
    # imageLoc_file = path + 'spatial/Spot_location.csv' #用不上
    class_file = path + 'Annotation_train_test_split.csv'

    # image_loc_in = pd.read_csv(imageLoc_file, header=0, index_col=0)#用不上
    image_rep_in = pd.read_csv(imageRep_file, header=0, index_col=0)
    rna_exp = pd.read_csv(RNA_file, header=0, index_col=0)
    class_data = pd.read_csv(class_file, header=0, index_col=0)
    features = np.array(rna_exp)
    # image_loc_in = image_loc_in.reindex(class_data.index)#需改
    labels = encode_onehot(class_data.Cluster.array)

    ##idx = rna_exp.index.T
    # idx_map = {j: i for i, j in enumerate(idx)}
    edge1 = cosine_similarity(image_rep_in)  # 余弦相似度
    ##edge_1 = edge1.flatten()
   # edge2 = 1-distance.cdist(image_rep_in, image_rep_in, metric='euclidean')  # 欧氏距离,transforms.normalize(mean_vals, std_vals)数据标准化
    edge3 = np.corrcoef(image_rep_in)#相关系数矩阵

    edge2=distance.cdist(image_rep_in, image_rep_in, metric='euclidean')
    edge2=sigmoid_function(edge2)
    #edge2[edge2>600]=0
    #edge2[edge2<600]=1
    ##edge_2 = edge2.flatten()
    # eage_value = (eage_1+eage_2)/2
    # eage_value = eage_value.flatten()#从第dim个维度开始展开，将后面的维度转化为一维.

    ##row_index = []
    ##col_index=[]
    ##for i in range(labels.shape[0]):
    ##for j in range (labels.shape[0]):
    ##row_index.append(i)
    ##col_index.append(j)

    ##adj1 = sp.coo_matrix((edge_1,(row_index,col_index)),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    # .coo_matrix(data,(row,col))  .onesnp.ones()函数返回给定形状和数据类型的新数组，# 其中元素的值设置为1。此函数与numpy zeros()函数非常相似。np.ones(shape, dtype=None, order='C')
    ##adj2 = sp.coo_matrix((edge_2, (row_index, col_index)), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    features = normalize_features(features)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj = adj + sp.eye(adj.shape[0]) #增加自连接

    idx_train = range(3000)
    idx_val = range(3000, 3500)
    idx_test = range(3500, 3639)

    adj1 = torch.FloatTensor(np.array(edge1))
    adj2 = torch.FloatTensor(np.array(edge2))
    adj3 = torch.FloatTensor(np.array(edge3))
    edge_attr = [adj1,adj2,adj3]
    # edge_attr =[adj,adj.t(),adj+adj.t()]
    edge_attr = torch.stack(edge_attr, dim=0)
    edge_attr = DSN(edge_attr)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return edge_attr, features, labels, idx_train, idx_val, idx_test



def DSN2(t):
    a=t.sum(dim=1,keepdim=True)
    b=t.sum(dim=0,keepdim=True)
    lamb=torch.cat([a.squeeze(),b.squeeze()],dim=0).max()
    r=t.shape[0]*lamb-t.sum(dim=0).sum(dim=0)
    
    a=a.expand(-1,t.shape[1])
    b=b.expand(t.shape[0],-1)
    tt=t+(lamb**2-lamb*(a+b)+a*b)/r

    ttmatrix=tt/tt.sum(dim=0)[0]
    ttmatrix=torch.where(t>0,ttmatrix,t)
    return ttmatrix


def DSN(x):
    """Doubly stochastic normalization"""
    p=x.shape[0]
    y1=[]
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1=torch.stack(y1,dim=0)
    return y1

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    """input is a numpy array""" 
    rowsum = mx.sum(axis=1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

