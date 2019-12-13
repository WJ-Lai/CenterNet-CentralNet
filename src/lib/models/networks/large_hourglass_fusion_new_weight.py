# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class MergeUp(nn.Module):
    def forward(self, *args):
        out = args[0]
        for layer in args:
            out = layer+out
        return out

def make_merge_layer(dim):
    return MergeUp()


# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)


class kp_module(nn.Module):
    def __init__(
            self, n, dims, modules, layer=residual,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
            make_low_layer(
                3, next_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)




class kp_module2(nn.Module):
    def __init__(
            self, n, dims, modules, layer=residual,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module2, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module2(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
            make_low_layer(
                3, next_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)

class kp_module3(nn.Module):
    def __init__(
            self, n, dims, modules, layer=residual,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module3, self).__init__()

        self.n = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module2(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
            make_low_layer(
                3, next_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2 = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)

class exkp(nn.Module):
    def __init__(
            self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256,
            make_tl_layer=None, make_br_layer=None,
            make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
            make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
            make_up_layer=make_layer, make_low_layer=make_layer,
            make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer,
            kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack = nstack
        self.heads = heads

        curr_dim = dims[0]

        self.pre1 = nn.Sequential(
            convolution(7, 3, 64, stride=2),
            residual(3, 64, 128, stride=2)
        ) if pre is None else pre

        self.kps1 = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs1 = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters1 = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu1 = nn.ReLU(inplace=True)


        # sensor2
        self.pre2 = nn.Sequential(
            convolution(7, 3, 64, stride=2),
            residual(3, 64, 128, stride=2)
        ) if pre is None else pre

        self.kps2 = nn.ModuleList([
            kp_module2(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
            ) for _ in range(nstack)
        ])

        self.cnvs2 = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters2 = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu2 = nn.ReLU(inplace=True)


        # sensor3
        self.pre3 = nn.Sequential(
            convolution(7, 3, 64, stride=2),
            residual(3, 64, 128, stride=2)
        ) if pre is None else pre

        self.kps3 = nn.ModuleList([
            kp_module3(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
            ) for _ in range(nstack)
        ])

        self.cnvs3 = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters3 = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)

        self.relu3 = nn.ReLU(inplace=True)

        self.mergeup_pre = make_weight_merge_layer(2)
        self.mergeup_kp = nn.ModuleList([
                            make_weight_merge_layer(3) for _ in range(nstack)
                            ])
        self.mergeup_head = nn.ModuleList([
                            make_weight_merge_layer(3) for _ in range(nstack)
                            ])

    def forward(self, image1, image2):

        inter1 = self.pre1(image1)
        inter2 = self.pre2(image2)
        inter3 = self.mergeup_pre(inter1, inter2)
        outs = []

        for ind in range(self.nstack):
            kp_1, cnv_1 = self.kps1[ind], self.cnvs1[ind]
            kp1 = kp_1(inter1)
            cnv1 = cnv_1(kp1)

            kp_2, cnv_2 = self.kps2[ind], self.cnvs2[ind]
            kp2 = kp_2(inter2)
            cnv2 = cnv_2(kp2)

            kp_3, cnv_3 = self.kps3[ind], self.cnvs3[ind]
            kp3 = kp_3(inter3)
            merge_kp = self.mergeup_kp[ind]
            kp3 = merge_kp(kp1, kp2, kp3)

            cnv3 = cnv_3(kp3)

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y1 = layer(cnv1)

                layer = self.__getattr__(head)[ind]
                y2 = layer(cnv2)

                layer = self.__getattr__(head)[ind]
                y3 = layer(cnv3)

                merge_head = self.mergeup_head[ind]
                out[head] = merge_head(y1, y2, y3)

            outs.append(out)
            if ind < self.nstack - 1:
                inter1 = self.inters_1[ind](inter1) + self.cnvs_1[ind](cnv1)
                inter1 = self.relu1(inter1)
                inter1 = self.inters1[ind](inter1)

                inter2 = self.inters_2[ind](inter2) + self.cnvs_2[ind](cnv2)
                inter2 = self.relu2(inter2)
                inter2 = self.inters2[ind](inter2)

                inter3 = self.inters_3[ind](inter3) + self.cnvs_3[ind](cnv3)
                inter3 = self.relu3(inter3)
                inter3 = self.inters3[ind](inter3)

        return outs


class WeightMergeUp(nn.Module):
    def __init__(self, num_input):
        super(WeightMergeUp, self).__init__()
        self.alpha = [torch.nn.Parameter(torch.Tensor([1.]).cuda()) for i in range(num_input)]

    def forward(self, *up):

        return sum([self.alpha[i]*up[i] for i in range(len(up))])
        # if len(up)==2:
        #     return self.alpha1 * up[0] + self.alpha2 * up[1]
        # else:
        #     return self.alpha1 * up[0] + self.alpha2 * up[1] + self.alpha3 * up[2]


def make_weight_merge_layer(num_input):
    return WeightMergeUp(num_input)


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet_fusion(exkp):
    def __init__(self, heads, num_stacks=2):
        n = 5
        # dims = [256, 256, 384, 384, 384, 512]
        dims = [128, 128, 192, 192, 192, 256]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet_fusion, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )
        self._name = 'fusion'


def get_large_hourglass_fusion_net(num_layers, heads, head_conv):
    model = HourglassNet_fusion(heads, 2)
    return model
