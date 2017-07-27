# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
import pickle

from sklearn.cross_validation import train_test_split


with open("cifar10_mini_train", "rb") as f:
    f = pickle.load(f)
    train_data = f["data"]
    train_label = f["labels"]



labels = [0 for i in range(10)]

y = []

for idx,num in enumerate(train_label):
    labels = [0 for i in range(10)]

    labels[num] = 1
    y.append(labels)


Y = np.array(y)
X = train_data

X = X.reshape(-1,3,32,32)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


network = DeepConvNet()
trainer = Trainer(network, X_train, Y_train, X_test, Y_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
