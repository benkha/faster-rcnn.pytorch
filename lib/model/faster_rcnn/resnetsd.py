from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''ResNet in PyTorch with stochastic depth

Repository: https://github.com/felixgwu/img_classification_pk_pytorch
Paper: https://arxiv.org/pdf/1603.09382.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.faster_rcnn.resnet_base_new import ResNet
from model.faster_rcnn.resnet_base_new import BasicBlock
from model.faster_rcnn.resnet_base_new import Bottleneck

from torch.distributions.bernoulli import Bernoulli


class Blocksd:

    def __init__(self, *args, total=1, index=0, last_layer_prob=0.5, **kwargs):
        """probability specified in https://arxiv.org/pdf/1603.09382.pdf"""
        super().__init__(*args, **kwargs)

        if self.args.stochastic_depth_mode == 'constant':
            prob = 0.75
        elif self.args.stochastic_depth_mode == 'layerReverse':
            prob = 1 - ((total - index) / (total - 1.)) * (1 - last_layer_prob)
        else:
            prob = 1 - (index / (total - 1.)) * (1 - last_layer_prob)

        self.p = Bernoulli(torch.tensor([prob]))

    def write(self, writer, layer, block, iteration):
        writer.add_scalar('layer{}/block{}/p'.format(layer, block), self.p.mean, iteration)

    def stochastic_forward(self, x, if_callback, else_callback):
        if self.p.sample() == 1 or not self.training:
            return if_callback(x)
        return else_callback(x)

    def forward(self, x):
        if_callback = super().forward
        def else_callback(x):
            out = self.shortcut(x)
            return F.relu(out)
        return self.stochastic_forward(x, if_callback, else_callback)


class BasicBlocksd(Blocksd, BasicBlock):
    pass


class Bottlenecksd(Blocksd, Bottleneck):
    pass


class ResNetsd(ResNet):

    def __init__(self, *args, **kwargs):
        self.global_index_block = 0
        self.total_num_blocks = sum(args[2])
        super().__init__(*args, **kwargs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.args, self.in_planes, planes, stride,
                index=self.global_index_block, total=self.total_num_blocks))
            self.in_planes = planes * block.expansion
            self.global_index_block += 1
        return nn.Sequential(*layers)

def ResNet18sd(args, num_classes=10):
    return ResNetsd(args, BasicBlocksd, [2,2,2,2], num_classes=num_classes)

def ResNet34sd(args, num_classes=10):
    return ResNetsd(args, BasicBlocksd, [3,4,6,3], num_classes=num_classes)

def ResNet50sd(args, num_classes=10):
    return ResNetsd(args, Bottlenecksd, [3,4,6,3], num_classes=num_classes)

def ResNet101sd(args, num_classes=10):
    return ResNetsd(args, Bottlenecksd, [3,4,23,3], num_classes=num_classes)

def ResNet152sd(args, num_classes=10):
    return ResNetsd(args, Bottlenecksd, [3,8,36,3], num_classes=num_classes)
