import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class ThaiNet(Chain):
    def __init__(self):
        super(ThaiNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3,
                out_channels=10,
                ksize=5,
                stride=1
            )
            self.conv2 = L.Convolution2D(
                in_channels=10,
                out_channels=20,
                ksize=5,
                stride=1
            )
            self.conv3 = L.Convolution2D(
                in_channels=20,
                out_channels=150,
                ksize=3,
                stride=1
            )
            self.fc1 = L.Linear(None, 400)
            self.fc2 = L.Linear(400, 86)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc1(h))
        if chainer.config.train:
            return self.fc2(h)
        return F.softmax(self.fc(h))
