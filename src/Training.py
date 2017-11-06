import numpy as np
import chainer
from chainer import cuda, Function, gradient_check
from chainer import report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples

from PNGDataset import PNGDataset
from PNGIterator import PNGIterator
from ThaiNet import ThaiNet

BATCH_SIZE = 150
PATH = './images/'
MAX_EPOCH = 10

dataset = PNGDataset(PATH)
featureFiles, labels = dataset.getDataset()

dataIterator = PNGIterator(featureFiles, labels, BATCH_SIZE)

model = ThaiNet()

optimizer = optimizers.Adam()
optimizer.setup(model)
batchCount = 0

while dataIterator.epoch < MAX_EPOCH:

        print('Training in epoch ' + repr(dataIterator.epoch) +
              ' on batch ' + repr(batchCount))
        trainBatch = dataIterator.next()
        imageTrain, targetTrain = concat_examples(trainBatch)

        predictionTrain = model(imageTrain)

        loss = F.softmax_cross_entropy(predictionTrain, targetTrain)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        batchCount += 1

        if dataIterator.is_new_epoch:

            print('epoch:{:02d} train_loss:{:.04f} '.format(
                  dataIterator.epoch, float(loss.data)), end='')
            batchCount = 0

serializers.save_npz('thaiNet.model', model)
