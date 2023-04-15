import numpy as np
import pandas as pd
import mxnet as mx
import logging
VALIDATION_SIZE = 2000
train = pd.read_csv('data/input/digit-recognizer/train.csv')
target = train['label'].values.ravel()
train = train.iloc[:, 1:].values

def get_lenet():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type='tanh')
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type='max', kernel=(2, 2), stride=(2, 2))
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type='tanh')
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type='max', kernel=(2, 2), stride=(2, 2))
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type='tanh')
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
val_data = train[:VALIDATION_SIZE].astype('float32')
val_label = target[:VALIDATION_SIZE]
train_data = train[VALIDATION_SIZE:, :].astype('float32')
train_label = target[VALIDATION_SIZE:]
train_data = np.array(train_data).reshape((-1, 1, 28, 28))
val_data = np.array(val_data).reshape((-1, 1, 28, 28))
train_data[:] /= 256.0
val_data[:] /= 256.0
batch_size = 500
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
devs = mx.gpu(0)
network = get_lenet()
model = mx.model.FeedForward(symbol=network, num_epoch=4, learning_rate=0.1, momentum=0.9, wd=1e-05, initializer=mx.init.Xavier(factor_type='in', magnitude=2.34))
eval_metrics = ['accuracy']