from __future__ import print_function

import os, sys

from chainer.links.caffe import CaffeFunction
from chainer import serializers

print('load VGG16 caffemodel')
vgg = CaffeFunction('pretrained_model/VGG_ILSVRC_16_layers.caffemodel')
print('save "vgg16.npz"')
serializers.save_npz('pretrained_model/vgg16.npz', vgg)
