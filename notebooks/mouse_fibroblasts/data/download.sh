#!/bin/bash

git clone https://github.com/sandberg-lab/txburst
mv txburst/data/mouse_gene_annotation.csv .
mv txburst/data/cell_cycle_annotation.csv .
mv txburst/data/SS3_c57_UMIs_concat.csv .
mv txburst/data/SS3_cast_UMIs_concat.csv .
rm -rf txburst
