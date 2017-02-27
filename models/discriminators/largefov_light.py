import os, sys

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
import functions as f


class LargeFOVLight(chainer.Chain):

    def __init__(self, n_class=21):
        super(LargeFOVLight, self).__init__(
            conv1_1=L.Convolution2D(3*n_class, 96, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(96, 128, 3, stride=1, pad=1),
            conv1_3=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 2, 3, stride=1, pad=1),
        )

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.conv1_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = self.conv3_2(h)
        h = f.global_average_pooling_2d(h)
        h = F.reshape(h, (h.shape[0],h.shape[1]))
        return h
