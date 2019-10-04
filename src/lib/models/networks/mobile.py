import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .resnet_dcn import fill_up_weights, fill_fc_weights, BN_MOMENTUM
import torch.utils.model_zoo as model_zoo


class hswish(nn.Module):
  def forward(self, x):
    out = x * F.relu6(x + 3, inplace=True) / 6
    return out


class hsigmoid(nn.Module):
  def forward(self, x):
    out = F.relu6(x + 3, inplace=True) / 6
    return out


class SeModule(nn.Module):
  def __init__(self, in_size, reduction=4):
    super(SeModule, self).__init__()
    self.se = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_size,
                  in_size // reduction,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False), nn.BatchNorm2d(in_size // reduction),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_size // reduction,
                  in_size,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False), nn.BatchNorm2d(in_size), hsigmoid())

  def forward(self, x):
    return x * self.se(x)


class Block(nn.Module):
  '''expand + depthwise + pointwise'''
  def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear,
               semodule, stride):
    super(Block, self).__init__()
    self.stride = stride
    self.se = semodule

    self.conv1 = nn.Conv2d(in_size,
                           expand_size,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(expand_size)
    self.nolinear1 = nolinear
    self.conv2 = nn.Conv2d(expand_size,
                           expand_size,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=kernel_size // 2,
                           groups=expand_size,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(expand_size)
    self.nolinear2 = nolinear
    self.conv3 = nn.Conv2d(expand_size,
                           out_size,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
    self.bn3 = nn.BatchNorm2d(out_size)

    self.shortcut = nn.Sequential()
    if stride == 1 and in_size != out_size:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_size,
                    out_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
          nn.BatchNorm2d(out_size),
      )

  def forward(self, x):
    out = self.nolinear1(self.bn1(self.conv1(x)))
    out = self.nolinear2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.se != None:
      out = self.se(out)
    out = out + self.shortcut(x) if self.stride == 1 else out
    return out


class MobileNetV3_Large(nn.Module):
  def __init__(self, num_classes=1000):
    super(MobileNetV3_Large, self).__init__()
    self.conv1 = nn.Conv2d(3,
                           16,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.hs1 = hswish()

    self.bneck = nn.Sequential(
        Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
        Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
        Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
        Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
        Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
        Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
        Block(3, 40, 240, 80, hswish(), None, 2),
        Block(3, 80, 200, 80, hswish(), None, 1),
        Block(3, 80, 184, 80, hswish(), None, 1),
        Block(3, 80, 184, 80, hswish(), None, 1),
        Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
        Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
        Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
        Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
        Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
    )

    self.conv2 = nn.Conv2d(160,
                           960,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(960)
    self.hs2 = hswish()
    self.linear3 = nn.Linear(960, 1280)
    self.bn3 = nn.BatchNorm1d(1280)
    self.hs3 = hswish()
    self.linear4 = nn.Linear(1280, num_classes)
    self.init_params()

  def init_params(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
          init.constant_(m.bias, 0)

  def forward(self, x):
    out = self.hs1(self.bn1(self.conv1(x)))
    out = self.bneck(out)
    out = self.hs2(self.bn2(self.conv2(out)))
    out = F.avg_pool2d(out, 7)
    out = out.view(out.size(0), -1)
    out = self.hs3(self.bn3(self.linear3(out)))
    out = self.linear4(out)
    return out


class MobileNetV3_Small(nn.Module):
  """ from https://github.com/xiaolai-sqlai/mobilenetv3 """
  def __init__(self, heads, head_conv=256, num_classes=1000):
    super(MobileNetV3_Small, self).__init__()
    self.inplanes = 96
    self.heads = heads
    self.deconv_with_bias = False
    self.conv1 = nn.Conv2d(3,
                           16,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.hs1 = hswish()

    self.bneck = nn.Sequential(
        Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
        Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
        Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
        Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
        Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
        Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
    )

    self.deconv_layers = self._make_deconv_layer(
        3,
        [256, 128, 64],
        [4, 4, 4],
    )

    for head in self.heads:
      classes = self.heads[head]
      if head_conv > 0:
        fc = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True))
        if 'hm' in head:
          fc[-1].bias.data.fill_(-2.19)
        else:
          fill_fc_weights(fc)
      else:
        fc = nn.Conv2d(64,
                       classes,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       bias=True)
        if 'hm' in head:
          fc.bias.data.fill_(-2.19)
        else:
          fill_fc_weights(fc)
      self.__setattr__(head, fc)

  def init_params(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
          init.constant_(m.bias, 0)

  def forward(self, x):
    out = self.hs1(self.bn1(self.conv1(x)))
    out = self.bneck(out)
    out = self.deconv_layers(out)
    ret = {}
    for head in self.heads:
      ret[head] = self.__getattr__(head)(out)
    return [ret]

  def _get_deconv_cfg(self, deconv_kernel, index):
    if deconv_kernel == 4:
      padding = 1
      output_padding = 0
    elif deconv_kernel == 3:
      padding = 1
      output_padding = 1
    elif deconv_kernel == 2:
      padding = 0
      output_padding = 0

    return deconv_kernel, padding, output_padding

  def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
    assert num_layers == len(num_filters), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    assert num_layers == len(num_kernels), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'

    layers = []
    for i in range(num_layers):
      kernel, padding, output_padding = \
          self._get_deconv_cfg(num_kernels[i], i)

      planes = num_filters[i]
      # fc = DCN(self.inplanes,
      #          planes,
      #          kernel_size=(3, 3),
      #          stride=1,
      #          padding=1,
      #          dilation=1,
      #          deformable_groups=1)
      fc = nn.Conv2d(self.inplanes,
                     planes,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     bias=False)
      fill_fc_weights(fc)
      up = nn.ConvTranspose2d(in_channels=planes,
                              out_channels=planes,
                              kernel_size=kernel,
                              stride=2,
                              padding=padding,
                              output_padding=output_padding,
                              bias=self.deconv_with_bias)
      fill_up_weights(up)

      layers.append(fc)
      layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
      layers.append(nn.ReLU(inplace=True))
      layers.append(up)
      layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
      layers.append(nn.ReLU(inplace=True))
      self.inplanes = planes

    return nn.Sequential(*layers)


def get_mobile_net(num_layers, heads, head_conv):
  net = MobileNetV3_Small(heads, head_conv)
  url = "https://raw.githubusercontent.com/zhen8838/Bin/master/mbv3_small-5889a514.pth"
  print('=> loading pretrained model {}'.format(url))
  pretrained_state_dict = model_zoo.load_url(url)
  net.load_state_dict(pretrained_state_dict, strict=False)

  return net


def test():
  heads = {'hm': 20, 'wh': 2, 'off': 2}
  head_conv = 256
  net = MobileNetV3_Small(heads, head_conv)

  model = torch.load("/home/zqh/Documents/CenterNet/mbv3_small-5889a514.pth",
                     map_location='cpu')
  weights = model["state_dict"]  # type:dict

  net.load_state_dict(weights, strict=False)

  return net
