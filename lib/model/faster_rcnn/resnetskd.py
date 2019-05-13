from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''ResNet in PyTorch with stochastic k-depth'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.faster_rcnn.resnetsd import ResNetsd
from model.faster_rcnn.resnetsd import BasicBlocksd
from model.faster_rcnn.resnetsd import Bottlenecksd
import numpy as np

import random


class Blockskd:

    def __init__(self, *args, index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.subsample = []

    def write(self, writer, layer, block, iteration):
        pass

    def set_subsample(self, subsample):
        self.subsample = subsample

    def stochastic_forward(self, x, if_callback, else_callback):
        if self.index in self.subsample or not self.training:
            return if_callback(x)
        return else_callback(x)


class BasicBlockskd(Blockskd, BasicBlocksd):
    pass


class Bottleneckskd(Blockskd, Bottlenecksd):
    pass


class ResNetskd(ResNetsd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def k(self):
        schedule = self.args.stochastic_depth_k_schedule
        total = self.args.epochs
        length = self.total_num_blocks - 1.
        start = self.args.stochastic_depth_start
        assert(self.args.stochastic_depth_power >= 0)

        if schedule == 'constant':
            return self.args.stochastic_depth_k
        if schedule == 'increaseLinear':
            fraction = self.epoch / total
            return int(round(fraction * length) + start)
        if schedule == 'decreaseLinear':
            fraction = 1. - (self.epoch / total)
            return int(round(fraction * length) + start)
        if schedule == 'increasePower':
            fraction = self.epoch / total
            power = self.args.stochastic_depth_power
            return int(round((fraction ** power) * length) + start)
        if schedule == 'decreasePower':
            fraction = 1. - (self.epoch / total)
            power = self.args.stochastic_depth_power
            return int(round((fraction ** power) * length) + start)

        raise UserWarning('Invalid stochastic k-depth mode: {}'.format(mode))

    @staticmethod
    def get_layer_sample_probabilities(slope, total):
        """Generate layer-wise sampling probabilities, for np.random.choice.

        Positive slope indicates bias towards sampling from end of network.
        Conversely, negative slope indicates bias towards the front.

        Note that bias is actually defined as slope * (x + bias).

        >>> ResNetskd.get_layer_sample_probabilities(-1, 3)
        array([0.5       , 0.33333333, 0.16666667])
        >>> ResNetskd.get_layer_sample_probabilities(1, 3)
        array([0.16666667, 0.33333333, 0.5       ])
        >>> ResNetskd.get_layer_sample_probabilities(0, 3)
        array([0.33333333, 0.33333333, 0.33333333])
        >>> small = ResNetskd.get_layer_sample_probabilities(0.25, 3)
        >>> big = ResNetskd.get_layer_sample_probabilities(25, 3)
        >>> np.std(small) < np.std(big)
        True
        """
        probs = np.array([slope * (x + 1) for x in range(total)])
        if any(probs < 0):
            probs -= probs.min() - 1
        if all(probs == 0):
            probs = np.array([1] * total)
        return probs / (probs.sum() + 1e-10)

    def forward(self, x, *args, **kwargs):
        p = self.get_layer_sample_probabilities(
            self.args.stochastic_depth_slope, self.total_num_blocks)
        subsample = np.random.choice(self.total_num_blocks, self.k(), p=p, replace=False)

        def callback(i, j, layer, block):
            if hasattr(block, 'set_subsample'):
                block.set_subsample(subsample)
        self.for_block_in_blocks(callback)
        return super().forward(x, *args, **kwargs)

    def write(self, writer, iteration):
        super().write(writer, iteration)
        writer.add_scalar('k', self.k(), iteration)

    def hook_iterate(self, epoch, batch_idx, global_step, optimizer):
        super().hook_iterate(epoch, batch_idx, global_step, optimizer)
        self.epoch = epoch


def ResNet18skd(args, num_classes=10):
    return ResNetskd(args, BasicBlockskd, [2,2,2,2], num_classes=num_classes)

def ResNet34skd(args, num_classes=10):
    return ResNetskd(args, BasicBlockskd, [3,4,6,3], num_classes=num_classes)

def ResNet50skd(args, num_classes=10):
    return ResNetskd(args, Bottleneckskd, [3,4,6,3], num_classes=num_classes)

def ResNet101skd(args, num_classes=10):
    return ResNetskd(args, Bottleneckskd, [3,4,23,3], num_classes=num_classes)

def ResNet152skd(args, num_classes=10):
    return ResNetskd(args, Bottleneckskd, [3,8,36,3], num_classes=num_classes)
