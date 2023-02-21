# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:33:02 2022

@author: dell
"""

# 导入一些需要的包
import os
import pandas as pd 
import numpy as np
import scanpy as sc
from graph_embedding import RNA_encoding_train
from pathlib import Path
from stMVC.utilities import parameter_setting
from sklearn.model_selection import train_test_split
from stMVC.utilities import normalize
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from stMVC.modules   import AE

def Preprocessing( args ):


	
	args.inputPath      = Path( args.basePath )
	
	args.outPath        = Path( args.basePath + 'stMVC/' ) 
	args.outPath.mkdir(parents=True, exist_ok=True)
    

adata = sc.read_csv("DATA.csv", first_column_names=False)

parser  =  parameter_setting()
args    =  parser.parse_args()
Preprocessing(args)
outDir = 'F:/liying/kongzhuan/2/'
test_size_prop=0.1

if test_size_prop > 0 :
	train_index, test_index = train_test_split(np.arange(adata.n_obs), 
												   test_size    = test_size_prop, 
												   random_state = 200)
else:
	train_index, test_index = list(range( adata.n_obs )), list(range( adata.n_obs ))
adata  = normalize( adata, filter_min_counts=False, size_factors=True,
						normalize_input=False, logtrans_input=True ) 
Nsample1, Nfeature1 =  np.shape( adata )
train           = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
												torch.from_numpy( adata[train_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][train_index].values ) )
train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size_T, shuffle = True )

test            = data_utils.TensorDataset( torch.from_numpy(adata[test_index].X ),
												torch.from_numpy( adata[test_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][test_index].values ) )
test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

total           = data_utils.TensorDataset( torch.from_numpy( adata.X ),
												torch.from_numpy( adata.obs['size_factors'].values ) )
total_loader    = data_utils.DataLoader( total, batch_size = args.batch_size_T, shuffle = False )

AE_structure = [Nfeature1, 1000, 50, 1000, Nfeature1]
model        = AE( [Nfeature1, 1000, 50], layer_d = [50, 1000], 
					   hidden1 = 1000, args = args, droprate = 0, type = "NB"  )

if args.use_cuda:
	model.cuda()

model.fit( train_loader, test_loader )

save_name_pre = '{}_{}-{}'.format( args.batch_size_T, args.lr_AET , '-'.join( map(str, AE_structure )) )
latent_z      = model.predict(total_loader, out='z' )

torch.save(model,'F:/liying/kongzhuan/2/'+ '/{}_RNA_AE_model.pt'.format(save_name_pre) )
latent_z1  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( 'F:/liying/kongzhuan/2/' + '/{}_RNA_AE_latent.csv'.format(save_name_pre) ) 



