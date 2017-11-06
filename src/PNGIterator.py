from __future__ import division

import numpy as np
import imageio

from chainer.dataset import iterator


class PNGIterator(iterator.Iterator):

    """Dataset iterator that serially reads PNG files as examples.

    This is an implementation of :class:`~chainer.dataset.Iterator`
    that reads features from PNG files and labels from a list in either the
    order of indexes or a shuffled order.

    This Iterator is adapted from :class:`~chainer.Iterators.SerialIterator`.

    Args:
        featureFiles: List of paths to PNG files.
        labels: List of labels for PNG files
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """

    def __init__(self, featureFiles, labels,
                 batch_size, repeat=True, shuffle=True):
        self.featureFiles = featureFiles
        self.labels = labels
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.labels)

        if self._order is None:
            if i_end <= N:
                toLoad = range(i, i_end)
            else:
                toLoad = range(i, N)
        else:
            toLoad = self._order[i:i_end]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    np.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        toLoad.extend(range(rest))
                    else:
                        np.append(toLoad, self._order[:rest])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        batch = []
        for index in toLoad:
            img = imageio.imread(
                    self.featureFiles[index]).astype(np.dtype('Float32'))
            img *= (1/img.max())
            img = np.transpose(img, (2, 0, 1))
            batch.append((img, self.labels[index]))

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.labels)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.labels)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        if self._shuffle:
            self._order = np.random.permutation(len(self.labels))
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
