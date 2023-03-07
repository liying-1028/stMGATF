# -*- coding: utf-8 -*-

import stlearn as st
import scanpy as sc
import numpy as np
import pandas as pd
import time
import os
import torch
import random
from pathlib import Path #路径

from stMGATF.utilities import parameter_setting#参数设置
from stMGATF.image_processing import tiling
from stMGATF.graph_embedding import RNA_encoding_train , Multi_views_attention_train


def Preprocessing( args ):

	start = time.time()
	
	args.inputPath      = Path( args.basePath )
	args.tillingPath    = Path( args.basePath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath        = Path( args.basePath + 'stMGATF/' ) 
	args.outPath.mkdir(parents=True, exist_ok=True)

	##load spatial transcriptomics and histological data
	adata  = sc.read_visium( args.inputPath )
	adata.var_names_make_unique()
	##过滤基因和细胞
	#sc.pp.filter_cells(adata, min_counts=3)
	#sc.pp.filter_genes(adata, min_cells=10)

	adata1 = adata.copy()

	sc.pp.normalize_total(adata, inplace=True)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
	adata2 = adata1[:, adata.var['highly_variable']]

	print('Successfully preprocessed {} genes and {} cells.'.format(adata2.n_vars, adata2.n_obs))

	args.use_cuda       = args.use_cuda and torch.cuda.is_available()

	## extract latent features of RNA-seq data by autoencoder-based framework
	print('Start training autoencoder-based framework for learning latent features')
	RNA_encoding_train(args, adata2, args.basePath + "stMGATF/")

	adata  = st.convert_scanpy(adata)
	#save physical location of spots into Spot_location.csv file
	data = { 'imagerow': adata.obs['imagerow'].values.tolist(), 'imagecol': adata.obs['imagecol'].values.tolist() }
	df   = pd.DataFrame(data, index = adata.obs_names.tolist())
	df.to_csv( args.basePath + 'spatial/' + 'Spot_location.csv' )

	##tilling histologicald data 
	print('Tilling spot image')
	tiling(adata, args.tillingPath, target_size = args.sizeImage)

	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":#主程序

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	Preprocessing(args)
