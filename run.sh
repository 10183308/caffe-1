make
make test 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib
make runtest GTEST_FILTER='DenseBlockLayerTest/*'

