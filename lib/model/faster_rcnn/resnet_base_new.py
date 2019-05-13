'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, args, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.args = args
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

  def write(self, writer, layer, block, iteration):
    pass


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, args, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.args = args
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

  def write(self, writer, layer, block, iteration):
    pass


class ResNet(nn.Module):
  def __init__(self, args, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64
    self.args = args

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change

    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.args, self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


  ##############
  # START DPAC #
  ##############

  def for_block_in_blocks(self, callback):
    results = []
    for i, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
      for j, block in enumerate(layer):
        result = callback(i, j, layer, block)
        if result is not None:
          results.append(result)
    return results

  def hook_iterate(self, epoch, batch_idx, global_step, optimizer):
    pass

  def write(self, writer, iteration):
    def callback(i, j, layer, block):
      if not hasattr(block, 'write'):
        return
      block.write(writer, i, j, iteration)
    return self.for_block_in_blocks(callback)

  def num_trainable_groups(self):
    return len([p for p in self.parameters() if p.requires_grad])

  def num_total_groups(self):
    return len(list(self.parameters()))

  ############
  # END DPAC #
  ############


def ResNet18(args, num_classes=10):
  return ResNet(args, BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(args, num_classes=10):
  return ResNet(args, BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(args, num_classes=10):
  return ResNet(args, Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(args, num_classes=10):
  return ResNet(args, Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(args, num_classes=10):
  return ResNet(args, Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
  net = ResNet18()
  y = net(torch.randn(1,3,32,32))
  print(y.size())

# test()
