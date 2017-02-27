#!/usr/bin/env python

import os, sys

import chainer

import utils

def main():
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
    dataset_dir = chainer.dataset.get_dataset_directory('pascal')
    path = os.path.join(dataset_dir, 'VOCtrainval_11-May-2012.tar')
    utils.cached_download(
        url,
        path=path,
        md5='6cd6e144f989b92b3379bac3b3de84fd',
    )
    utils.extract_file(path, to_directory=dataset_dir)


if __name__ == '__main__':
    main()
