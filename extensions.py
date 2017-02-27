import copy

import chainer.functions as F
from chainer import Variable
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extensions


class TestModeEvaluator(extensions.Evaluator):

    def __init__(self, iterator, updater, converter=convert.concat_examples,
                 device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(updater.model, link.Link):
            self._targets = {'main': updater.model}
        else:
            self._targets = updater.model

        self.updater = updater
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self.get_iterator('main')
        all_targets = self.get_all_targets()
        for model in all_targets.values():
            if hasattr(model, 'train'):
                model.train = False

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                self.updater.forward(batch)
                self.updater.calc_loss()
            summary.add(observation)

        for model in all_targets.values():
            if hasattr(model, 'train'):
                model.train = True
        return summary.compute_mean()
