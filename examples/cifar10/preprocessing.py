import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
import cv2
import sys

direct = 'examples/cifar10/'
db_train = leveldb.LevelDB(direct+'cifar10_train_leveldb')
db_test = leveldb.LevelDB(direct+'cifar10_test_leveldb')
datum = caffe_pb2.Datum()

#index = sys.argv[1]
size_train = 50000
size_test = 10000
data_train = np.zeros((size_train, 3, 32, 32))
label_train = np.zeros(size_train, dtype=int)

data_test = np.zeros((size_test, 3, 32, 32))
label_test = np.zeros(size_test, dtype=int)

print 'Reading training data...'
i = -1
for key, value in db_train.RangeIter():
    i = i + 1
    if i % 1000 == 0:
        print i
    if i == size_train:
        break
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data_train[i] = data
    label_train[i] = label


print 'Reading test data...'
i = -1
for key, value in db_test.RangeIter():
    i = i + 1
    if i % 1000 == 0:
        print i
    if i ==size_test:
        break
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data_test[i] = data
    label_test[i] = label


print 'Computing statistics...'
mean = np.mean(data_train, axis=(0,2,3))
std = np.std(data_train, axis=(0,2,3))

print mean

# np.savetxt('mean_cifar10.txt', mean)
# np.savetxt('std_cifar10.txt', std)

print 'Normalizing'
for i in range(3):
	print i
	data_train[:, i, :, :] = data_train[:, i, :, :] - mean[i]
	data_train[:, i, :, :] = data_train[:, i, :, :]/std[i]
	data_test[:, i, :, :] = data_test[:, i, :, :] - mean[i]
	data_test[:, i, :, :] = data_test[:, i, :, :]/std[i]

#Zero Padding
print 'Padding...'
npad = ((0,0), (0,0), (4,4), (4,4))
data_train = np.pad(data_train, pad_width=npad, mode='constant', constant_values=0)
data_test = np.pad(data_test, pad_width=npad, mode='constant', constant_values=0)

print 'Outputting training data'
leveldb_file = direct + 'cifar10_train_leveldb_padding'
batch_size = size_train

# create the leveldb file
db = leveldb.LevelDB(leveldb_file)
batch = leveldb.WriteBatch()
datum = caffe_pb2.Datum()

for i in range(size_train):
    if i % 1000 == 0:
        print i

    # save in datum
    datum = caffe.io.array_to_datum(data_train[i], label_train[i])
    keystr = '{:0>5d}'.format(i)
    batch.Put( keystr, datum.SerializeToString() )

    # write batch
    if(i + 1) % batch_size == 0:
        db.Write(batch, sync=True)
        batch = leveldb.WriteBatch()
        print (i + 1)

# write last batch
if (i+1) % batch_size != 0:
    db.Write(batch, sync=True)
    print 'last batch'
    print (i + 1)

print 'Outputting test data'
leveldb_file = direct + 'cifar10_test_leveldb_padding'
batch_size = size_test

# create the leveldb file
db = leveldb.LevelDB(leveldb_file)
batch = leveldb.WriteBatch()
datum = caffe_pb2.Datum()

for i in range(size_test):
    # save in datum
    datum = caffe.io.array_to_datum(data_test[i], label_test[i])
    keystr = '{:0>5d}'.format(i)
    batch.Put( keystr, datum.SerializeToString() )

    # write batch
    if(i + 1) % batch_size == 0:
        db.Write(batch, sync=True)
        batch = leveldb.WriteBatch()
        print (i + 1)

# write last batch
if (i+1) % batch_size != 0:
    db.Write(batch, sync=True)
    print 'last batch'
    print (i + 1)







