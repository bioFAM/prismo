#!/bin/bash

mkdir data
pushd data
wget https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5
wget https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_spatial.tar.gz
tar -xzf V1_Mouse_Brain_Sagittal_Anterior_spatial.tar.gz
rm V1_Mouse_Brain_Sagittal_Anterior_spatial.tar.gz
popd
