library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(scry)

if(!dir.exists(file.path("data"))){
    dir.create(file.path("data"))
}

InstallData("ssHippo")
slide.seq <- LoadData("ssHippo")

X <- slide.seq@images[[1]]@coordinates
slide.seq <- AddMetaData(slide.seq, X)
Y <- slide.seq@assays[[1]]@counts

dev <- devianceFeatureSelection(Y, fam="poisson")
dev[is.na(dev)] <- 0
slide.seq@assays[[1]] <- AddMetaData(slide.seq@assays[[1]], dev, col.name="deviance_poisson")
o <- order(dev,decreasing=TRUE)

dfile <- file.path("data/sshippo.h5Seurat")
SaveH5Seurat(slide.seq, filename=dfile)
Convert(dfile, dest="h5ad")
