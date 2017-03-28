mlton -cc-opt "-I./OpenBLAS -L./OpenBLAS/ -lopenblas" -link-opt "-I./OpenBLAS -L./OpenBLAS/ -lopenblas" -verbose 1 -export-header export.h testMatrix.mlb matrix.c
export LD_LIBRARY_PATH="./OpenBLAS"
sudo ldconfig
./testMatrix
