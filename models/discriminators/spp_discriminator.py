import os, sys

import chainer
import chainer.functions as F
import chainer.links as L


class SPPDiscriminator(chainer.Chain):

    def __init__(self, n_class=21):
        super(SPPDiscriminator, self).__init__(
            conv1_1=L.Convolution2D(3*n_class, 96, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(96, 128, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_1=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            fc4=L.Linear(in_size=None, out_size=64),
            fc5=L.Linear(in_size=None, out_size=32),
            fc6=L.Linear(in_size=None, out_size=2),
        )
        self.train=True

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.tanh(self.fc4(h))
        h = F.dropout(h, ratio=.5, train=self.train)
        h = F.tanh(self.fc5(h))
        h = F.dropout(h, ratio=.5, train=self.train)
        h = self.fc6(h)
        return h
