rm -r TClogFwd
rm -r TClogBwd
rm -r TC_TrueFwdlog
rm -r TC_TrueBwdlog
make -j64
make test -j64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib
make runtest GTEST_FILTER='DenseBlockLayerTest/*'

