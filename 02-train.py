#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

import dataset
from models.vgg16 import VGG16
from models.generators import FCN32s, FCN16s, FCN8s 
from models.discriminators import (
    LargeFOV, LargeFOVLight, SmallFOV, SmallFOVLight, SPPDiscriminator)
from updater import GANUpdater, NonAdversarialUpdater
from extensions import TestModeEvaluator
import utils


def parse_args(generators, discriminators, updaters):
    parser = argparse.ArgumentParser(description='Semantic Segmentation using Adversarial Networks')
    parser.add_argument('--generator', choices=generators.keys(), default='fcn32s',
                        help='Generator(segmentor) architecture')
    parser.add_argument('--discriminator', choices=discriminators.keys(), default='largefov',
                        help='Discriminator architecture')
    parser.add_argument('--updater', choices=updaters.keys(), default='gan',
                        help='Updater')
    parser.add_argument('--initgen_path', default='pretrained_model/vgg16.npz',
                        help='Pretrained model of generator')
    parser.add_argument('--initdis_path', default=None,
                        help='Pretrained model of discriminator')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=100000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='snapshot',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--evaluate_interval', type=int, default=1000,
                        help='Interval of evaluation')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    return parser.parse_args()

def load_pretrained_model(initmodel_path, initmodel, model, n_class, device):
    print('Initializing the model')
    chainer.serializers.load_npz(initmodel_path, initmodel)
    utils.copy_chainermodel(initmodel, model)
    return model

def make_optimizer(model, lr=1e-10, momentum=0.99):
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005), 'hook_dec')
    return optimizer

def main():
    generators = {
        'fcn32s': (FCN32s, VGG16, 1e-10), # (model, initmodel, learning_rate)
        'fcn16s': (FCN16s, FCN32s, 1e-12),
        'fcn8s': (FCN8s, FCN16s, 1e-14),
    }
    discriminators = {
        'largefov': (LargeFOV, LargeFOV, 0.1, 1.0), # (model, initmodel, learning_rate, L_bce_weight)
        'largefov-light': (LargeFOVLight, LargeFOVLight, 0.1, 1.0),
        'smallfov': (SmallFOV, SmallFOV, 0.1, 0.1),
        'smallfov-light': (SmallFOVLight, SmallFOVLight, 0.2, 1.0),
        'sppdis': (SPPDiscriminator, SPPDiscriminator, 0.1, 1.0),
    }
    updaters = {
        'gan': GANUpdater,
        'standard': NonAdversarialUpdater
    }

    args = parse_args(generators, discriminators, updaters)

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# iteration: {}'.format(args.iteration))

    # dataset
    train = dataset.PascalVOC2012Dataset('train')
    val = dataset.PascalVOC2012Dataset('val')
    n_class = len(train.label_names)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

    # Set up a neural network to train and an optimizer
    if args.updater=='gan':
        gen_cls, initgen_cls, lr = generators[args.generator]
        dis_cls, initdis_cls, lr, L_bce_weight = discriminators[args.discriminator]
        print('# generator: {}'.format(gen_cls.__name__))
        print('# discriminator: {}'.format(dis_cls.__name__))
        print('')

        # Initialize generator
        if args.initgen_path:
            gen, initgen = gen_cls(n_class), initgen_cls(n_class)
            gen = load_pretrained_model(args.initgen_path, initgen, gen, n_class, args.gpu)
        else:
            gen = gen_cls(n_class)
        # Initialize discriminator
        if args.initdis_path:
            dis, initdis = dis_cls(n_class), initdis_cls(n_class)
            dis = load_pretrained_model(args.initdis_path, initdis, dis, n_class, args.gpu)
        else:
            dis = dis_cls(n_class)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            gen.to_gpu()  # Copy the model to the GPU
            dis.to_gpu()
        opt_gen = make_optimizer(gen, lr)
        opt_dis = make_optimizer(dis, lr)
        model={'gen':gen,'dis':dis}
        optimizer={'gen': opt_gen, 'dis': opt_dis}
    elif args.updater=='standard':
        model_cls, initmodel_cls, lr = generators[args.generator]
        L_bce_weight = None
        print('# model: {}'.format(model_cls.__name__))
        print('')
        if args.initgen_path:
            model, initmodel = model_cls(n_class), initmodel_cls(n_class)
            model = load_pretrained_model(args.initgen_path, initmodel, model, n_class, args.gpu)
        else:
            model = model_cls(n_class)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            model.to_gpu()  # Copy the model to the GPU
        optimizer = make_optimizer(model, lr)

    # Set up a trainer
    updater = updaters[args.updater](
        model=model,
        iterator=train_iter,
        optimizer=optimizer,
        device=args.gpu,
        L_bce_weight=L_bce_weight,
        n_class=n_class,)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    evaluate_interval = (args.evaluate_interval, 'iteration') 
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')

    trainer.extend(
        TestModeEvaluator(
            val_iter, updater, device=args.gpu),
        trigger=snapshot_interval,
        invoke_before_training=True)
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    if args.updater=='gan':
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
        trainer.extend(extensions.LogReport(trigger=display_interval))
        trainer.extend(extensions.PrintReport([
            'iteration',
            'gen/loss', 'validation/gen/loss',
            'dis/loss',
            'gen/accuracy', 'validation/gen/accuracy',
            'gen/iu', 'validation/gen/iu',
            'elapsed_time',
        ]), trigger=display_interval)
    elif args.updater=='standard':
        trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
        trainer.extend(extensions.LogReport(trigger=display_interval))
        trainer.extend(extensions.PrintReport([
            'iteration',
            'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy',
            'main/iu', 'validation/main/iu',
            'elapsed_time',
        ]), trigger=display_interval)

    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    print('\nRun the training')
    trainer.run()

if __name__ == '__main__':
    main()
