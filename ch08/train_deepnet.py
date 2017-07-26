# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer


from sklearn.cross_validation import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


print(unpickle("data_batch_2")[b'data'].shape)
print(len(unpickle("data_batch_2")[b'labels']))

labels = [0 for i in range(10)]

y = []

for idx,num in enumerate(unpickle("data_batch_2")[b'labels']):
    labels = [0 for i in range(10)]

    labels[num] = 1
    y.append(labels)


Y = np.array(y)
X = unpickle("data_batch_2")[b'data']

X = X.reshape(-1,3,32,32)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


network = DeepConvNet()
trainer = Trainer(network, X_train, Y_train, X_test, Y_test,
                  epochs=50, mini_batch_size=50,
                  optimizer='AdaGrad', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
