mlton -cc-opt "-I../matrix/OpenBLAS -L../matrix/OpenBLAS -lopenblas" -link-opt "-I../matrix/OpenBLAS -L../matrix/OpenBLAS -lopenblas" -verbose 1 -export-header export.h test_NN.mlb ../matrix/matrix.c
export LD_LIBRARY_PATH="../matrix/OpenBLAS"
sudo ldconfig
./test_NN
