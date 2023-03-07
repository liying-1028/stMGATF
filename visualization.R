
rm(list=ls())
plot_colors=c("1" = "#6D1A9C", "2" = "#CC79A7","3"  = "#7495D3", "4" = "#59BE86", "5" = "#56B4E9", "6" = "#FEB915", 
              "7" = "#DB4C6C", "8" = "#C798EE", "9" = "#3A84E6", "10"= "#FF0099FF", "11" = "#CCFF00FF",
              "12" = "#268785", "13"= "#FF9900FF", "14"= "#33FF00FF", "15"= "#AF5F3C", "16"= "#DAB370", 
              "17" = "#554236", "18"= "#787878", "19"= "#877F6C","0"= "#F56867")



Seurat_processing = function(basePath, robust_rep, nDim = 20, nCluster = 20, save_path = NULL, pdf_file = NULL ){
  
  library("Seurat")
  library('ggplot2')
  
  idc = Load10X_Spatial(data.dir= basePath )
  idc = SCTransform(idc, assay = "Spatial", verbose = FALSE)
  idc = RunPCA(idc, assay = "SCT", verbose = FALSE)
  
  input_features = as.matrix(robust_rep[match(colnames(idc), row.names(robust_rep)),])
  original_pca_emb           = idc@reductions$pca@cell.embeddings
  row.names(input_features)  = row.names(original_pca_emb)
  idc@reductions$pca@cell.embeddings[,1:nDim] = input_features
  
  idc = FindNeighbors(idc, reduction = "pca", dims = 1:nDim)
  
  for(qq in seq(0.05,1.5,0.01))
  {
    idc <- FindClusters( idc, resolution = qq,  verbose = FALSE )
    if(length(table(Idents(idc)))==nCluster)
    {
      break
    }
  }
  
  idc = RunUMAP(idc, reduction = "pca", dims = 1:nDim)
  idc[["clusterings"]] = as.character(as.numeric(as.character(Idents(idc)))+1)
  idc[["clusterings"]] = idx_lei['P']
  pdf( paste0( save_path, pdf_file ), width = 10, height = 10)
  
  p1  = DimPlot(idc, reduction = "umap", label = T, label.size = 6, pt.size=1.5,
                cols = plot_colors, group.by = "clusterings")+
    theme(legend.position = "none",
          legend.title = element_blank())+
    ggtitle("")
  print(p1)
  
  p2  = SpatialDimPlot(idc, label = T, label.size = 3, cols = plot_colors, 
                       group.by = "clusterings" )+
    theme(legend.position = "none",
          legend.title = element_blank())+
    ggtitle("")
  
  print(p2)
  dev.off()
  
  return(idc)
}

robust_rep     = read.csv( "feature.csv", header = T, row.names = 1)
idx_lei=read.csv("pred.csv", header = T, row.names = 1)
Seurat_obj     = Seurat_processing(basePath, robust_rep, 20, 20, basePath, "stMGATF_Clustering.pdf" )



