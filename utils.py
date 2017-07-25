#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cStringIO as StringIO
except:
    from io import StringIO
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import six

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer import cuda
from chainer.training import extensions


# -----------------------------------------------------------------------------
# CV Util
# -----------------------------------------------------------------------------

def resize_img_with_max_size(img, max_size=500*500):
    """Resize image with max size (height x width)"""
    from skimage.transform import rescale
    height, width = img.shape[:2]
    scale = max_size / (height * width)
    resizing_scale = 1
    if scale < 1:
        resizing_scale = np.sqrt(scale)
        img = rescale(img, resizing_scale, preserve_range=True)
        img = img.astype(np.uint8)
    return img, resizing_scale


# -----------------------------------------------------------------------------
# Chainer Util
# -----------------------------------------------------------------------------

def copy_chainermodel(src, dst):
    from chainer import link
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    print('Copying layers %s -> %s:' %
          (src.__class__.__name__, dst.__class__.__name__))
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, link.Chain):
            copy_chainermodel(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch.' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print(' layer: %s -> %s' % (child.name, dst_child.name))



# -----------------------------------------------------------------------------
# Data Util
# -----------------------------------------------------------------------------

def download(url, path, quiet=False):

    def is_google_drive_url(url):
        m = re.match('^https?://drive.google.com/uc\?id=.*$', url)
        return m is not None

    if is_google_drive_url(url):
        client = 'gdown'
    else:
        client = 'wget'

    cmd = '{client} {url} -O {path}'.format(client=client, url=url, path=path)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))

    return path

def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5, quiet=False):
        if not quiet:
            print('Checking md5 of file: {}'.format(path))
        is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
        return is_same

    if os.path.exists(path) and not md5:
        return path
    elif os.path.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return download(url, path, quiet=quiet)


def extract_file(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar'):
        opener, mode = tarfile.open, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError("Could not extract '%s' as no appropriate "
                         "extractor is found" % path)

    cwd = os.getcwd()
    os.chdir(to_directory)
    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)



# -----------------------------------------------------------------------------
# Color Util
# -----------------------------------------------------------------------------

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(N=256):
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_true, label_pred, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = _fast_hist(label_true.flatten(), label_pred.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def draw_label(label, img, n_class, label_titles, bg_label=0):
    """Convert label to rgb with label titles.

    @param label_title: label title for each labels.
    @type label_title: dict
    """
    from PIL import Image
    from scipy.misc import fromimage
    from skimage.color import label2rgb
    from skimage.transform import resize
    colors = labelcolormap(n_class)
    label_viz = label2rgb(label, img, colors=colors[1:], bg_label=bg_label)
    # label 0 color: (0, 0, 0, 0) -> (0, 0, 0, 255)
    label_viz[label == 0] = 0

    # plot label titles on image using matplotlib
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    # plot image
    plt.imshow(label_viz)
    # plot legend
    plt_handlers = []
    plt_titles = []
    for label_value in np.unique(label):
        if label_value not in label_titles:
            continue
        fc = colors[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_titles[label_value])
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=0.5)
    # convert plotted figure to np.ndarray
    f = StringIO.StringIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    result_img_pil = Image.open(f)
    result_img = fromimage(result_img_pil, mode='RGB')
    result_img = resize(result_img, img.shape, preserve_range=True)
    result_img = result_img.astype(img.dtype)
    return result_img


def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical+h,
               pad_horizontal:pad_horizontal+w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, 3), dtype=np.uint8)
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y*one_height:(y+1)*one_height,
                                   x*one_width:(x+1)*one_width, ] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    from skimage.transform import resize

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return x_num, y_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(img, (h, w), preserve_range=True).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)

