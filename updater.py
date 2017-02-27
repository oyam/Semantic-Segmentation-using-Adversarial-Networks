#!/usr/bin/env python

from __future__ import print_function

import six
import numpy as np
import chainer
from chainer import link
import chainer.functions as F
from chainer import Variable

import utils


class UpdaterMixin(object):
    """Updater mixin class

    To use extensions.TestModeEvaluator, forward and calc_loss must be inplemented.
    """
    
    def forward(self, batch):
        raise NotImplementedError

    def calc_loss(self):
        raise NotImplementedError

    def _standard_updater_kwargs(self, **kwargs):
        standard_updater_args = [
            'iterator',
            'optimizer',
            'converter',
            'device',
            'loss_func']
        for key in kwargs.keys():
            if not key in standard_updater_args:
                del kwargs[key]
        return kwargs

class GANUpdater(chainer.training.StandardUpdater, UpdaterMixin):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model') # set for exeptions.Evaluator
        self.gen, self.dis = self.model['gen'], self.model['dis'] 
        self.L_bce_weight = kwargs.pop('L_bce_weight')
        self.n_class = kwargs.pop('n_class')
        self.xp = chainer.cuda.cupy if kwargs['device'] >= 0 else np
        kwargs = self._standard_updater_kwargs(**kwargs)
        super(GANUpdater, self).__init__(*args, **kwargs)

    def _get_loss_dis(self):
        batchsize = self.y_fake.data.shape[0]
        loss = F.softmax_cross_entropy(self.y_real, Variable(self.xp.ones(batchsize, dtype=self.xp.int32), volatile=not self.gen.train))
        loss += F.softmax_cross_entropy(self.y_fake, Variable(self.xp.zeros(batchsize, dtype=self.xp.int32), volatile=not self.gen.train))
        chainer.report({'loss': loss}, self.dis)
        return loss

    def _get_loss_gen(self):
        batchsize = self.y_fake.data.shape[0]
        L_mce = F.softmax_cross_entropy(self.pred_label_map, self.ground_truth, normalize=False)
        L_bce = F.softmax_cross_entropy(self.y_fake, Variable(self.xp.ones(batchsize, dtype=self.xp.int32), volatile=not self.gen.train))
        loss = L_mce + self.L_bce_weight * L_bce

        # log report
        label_true = chainer.cuda.to_cpu(self.ground_truth.data)
        label_pred = chainer.cuda.to_cpu(self.pred_label_map.data).argmax(axis=1)
        logs = []
        for i in six.moves.range(batchsize):
            acc, acc_cls, iu, fwavacc = utils.label_accuracy_score(
                label_true[i], label_pred[i], self.n_class)
            logs.append((acc, acc_cls, iu, fwavacc))
        log = np.array(logs).mean(axis=0)
        values = {
            'loss': loss,
            'accuracy': log[0],
            'accuracy_cls': log[1],
            'iu': log[2],
            'fwavacc': log[3],
        }
        chainer.report(values, self.gen)

        return loss

    def _make_dis_input(self, input_img, label_map):
        b = F.broadcast_to(input_img[:,0,:,:], shape=label_map.shape)
        g = F.broadcast_to(input_img[:,1,:,:], shape=label_map.shape)
        r = F.broadcast_to(input_img[:,2,:,:], shape=label_map.shape)
        product_b = label_map * b
        product_g = label_map * g
        product_r = label_map * r
        dis_input = F.concat([product_b, product_g, product_r], axis=1)
        return dis_input

    def _onehot_encode(self, label_map):
        for i, c in enumerate(six.moves.range(self.n_class)):
            mask = label_map==c
            mask = mask.reshape(1,mask.shape[0],mask.shape[1])
            if i==0:
                onehot = mask
            else:
                onehot = np.concatenate([onehot, mask]) 
        return onehot.astype(self.xp.float32)

    def forward(self, batch):
        label_onehot_batch = [self._onehot_encode(pair[1]) for pair in batch]

        input_img, ground_truth = self.converter(batch, self.device)
        ground_truth_onehot = self.converter(label_onehot_batch, self.device)
        input_img = Variable(input_img, volatile=not self.gen.train)
        ground_truth = Variable(ground_truth, volatile=not self.gen.train)
        ground_truth_onehot = Variable(ground_truth_onehot, volatile=not self.gen.train)
        
        x_real = self._make_dis_input(input_img, ground_truth_onehot)
        y_real = self.dis(x_real)

        pred_label_map = self.gen(input_img)
        x_fake = self._make_dis_input(input_img, F.softmax(pred_label_map))
        y_fake = self.dis(x_fake)

        self.y_fake = y_fake
        self.y_real = y_real
        self.pred_label_map = pred_label_map
        self.ground_truth = ground_truth
        
    def calc_loss(self):
        self.loss_dis = self._get_loss_dis()
        self.loss_gen = self._get_loss_gen()
        
    def backprop(self):
        self.dis.cleargrads()
        self.gen.cleargrads()
        self.loss_dis.backward()
        self.loss_gen.backward()
        self.get_optimizer('dis').update()
        self.get_optimizer('gen').update()

    def update_core(self):
        batch = self.get_iterator('main').next()
        self.forward(batch)
        self.calc_loss()
        self.backprop()


class NonAdversarialUpdater(chainer.training.StandardUpdater, UpdaterMixin):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model') # set for exeptions.Evaluator
        self.n_class = kwargs.pop('n_class')
        self.xp = chainer.cuda.cupy if kwargs['device'] >= 0 else np
        kwargs = self._standard_updater_kwargs(**kwargs)
        super(NonAdversarialUpdater, self).__init__(*args, **kwargs)

    def forward(self, batch):
        input_img, ground_truth = self.converter(batch, self.device)
        input_img = Variable(input_img, volatile=not self.model.train)
        self.ground_truth = Variable(ground_truth, volatile=not self.model.train)
        self.pred_label_map = self.model(input_img)
        
    def calc_loss(self):
        batchsize = self.ground_truth.shape[0]
        self.loss = F.softmax_cross_entropy(self.pred_label_map, self.ground_truth, normalize=False)

        # log report
        label_true = chainer.cuda.to_cpu(self.ground_truth.data)
        label_pred = chainer.cuda.to_cpu(self.pred_label_map.data).argmax(axis=1)
        logs = []
        for i in six.moves.range(batchsize):
            acc, acc_cls, iu, fwavacc = utils.label_accuracy_score(
                label_true[i], label_pred[i], self.n_class)
            logs.append((acc, acc_cls, iu, fwavacc))
        log = np.array(logs).mean(axis=0)
        values = {
            'loss': self.loss,
            'accuracy': log[0],
            'accuracy_cls': log[1],
            'iu': log[2],
            'fwavacc': log[3],
        }
        chainer.report(values, self.model)
        
    def backprop(self):
        self.model.cleargrads()
        self.loss.backward()
        self.get_optimizer('main').update()

    def update_core(self):
        batch = self.get_iterator('main').next()
        self.forward(batch)
        self.calc_loss()
        self.backprop()
