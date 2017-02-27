#!/usr/bin/env python

from __future__ import division

import argparse
import os

import chainer
import scipy.misc
import numpy as np

from models.generators import FCN32s, FCN16s, FCN8s 
import utils
from dataset import PascalVOC2012Dataset


class Inferencer(object):

    def __init__(self, dataset, model, gpu):
        self.dataset = dataset
        self.gpu = gpu
        self.model = model

        self.label_names = self.dataset.label_names

        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def _infer(self, x):
        self.model.train = False
        score = chainer.cuda.to_cpu(self.model(x).data)[0]
        label = np.argmax(score, axis=0)
        return label

    def infer_image_file(self, img_file):
        print('{0}:'.format(os.path.realpath(img_file)))
        # setup input
        img = scipy.misc.imread(img_file, mode='RGB')
        img, resizing_scale = utils.resize_img_with_max_size(img)
        print(' - resizing_scale: {0}'.format(resizing_scale))
        datum = self.dataset.img_to_datum(img.copy())
        x_data = np.array([datum], dtype=np.float32)
        if self.gpu >= 0:
            x_data = chainer.cuda.to_gpu(x_data, device=self.gpu)
        x = chainer.Variable(x_data, volatile=False)
        label = self._infer(x)
        return img, label

    def visualize_label(self, img, label):
        # visualize result
        unique_labels, label_counts = np.unique(label, return_counts=True)
        print('- labels:')
        label_titles = {}
        for label_value, label_count in zip(unique_labels, label_counts):
            label_region = label_count / label.size
            if label_region < 0.001:
                continue
            title = '{0}:{1} = {2:.1%}'.format(
                label_value, self.label_names[label_value], label_region)
            label_titles[label_value] = title
            print('  - {0}'.format(title))
        labelviz = utils.draw_label(
            label, img, n_class=len(self.label_names),
            label_titles=label_titles)
        # save result
        return utils.get_tile_image([img, labelviz])


def main():
    segmentors = {
        'fcn32s': FCN32s,
        'fcn16s': FCN16s,
        'fcn8s': FCN8s,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-s', '--segmentor', choices=segmentors.keys(), default='fcn32s',
                        help='Segmentor arch')
    parser.add_argument('-w', '--weight',
                        help='Pretrained model of segmentor')
    parser.add_argument('--n_class', default=21, type=int,
                        help='number of classes')
    parser.add_argument('-i', '--img-files', nargs='+', required=True,
                        help='path to image files')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    img_files = args.img_files
    gpu = args.gpu
    model = segmentors[args.segmentor](args.n_class)
    print('load initmodel..')
    chainer.serializers.load_npz(args.weight, model)
    save_dir = args.out
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = PascalVOC2012Dataset('val')

    infer = Inferencer(dataset, model, gpu)
    for img_file in img_files:
        img, label = infer.infer_image_file(img_file)
        out_img = infer.visualize_label(img, label)

        out_file = os.path.join(save_dir, os.path.basename(img_file))
        scipy.misc.imsave(out_file, out_img)
        print('- out_file: {0}'.format(out_file))


if __name__ == '__main__':
    main()
