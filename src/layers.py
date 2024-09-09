# This Python file is built on top of the MTGNN project (https://github.com/nnzhan/MTGNN).
# The original work is described in the following paper:
#
# @inproceedings{wu2020connecting,
#   title={Connecting the dots: Multivariate time series forecasting with graph neural networks},
#   author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
#   booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining},
#   pages={753--763},
#   year={2020}
# }
#
# This file extends and modifies the original MTGNN framework to suit CGAD framework.

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import GCNConv
import numbers
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, c_in, c_out, dropout):
        super(GCN, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout

        self.conv1 = GCNConv(self.c_in, 2 * self.c_in, cached=True)
        self.conv2 = GCNConv(2 * self.c_in, self.c_out, cached=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        return x


class GCN_dcaug(torch.nn.Module):
    def __init__(self, c_in, c_out, dropout):
        super(GCN_dcaug, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout

        self.conv1 = GCNConv(self.c_in, 2 * self.c_in, cached=True)
        self.conv2 = GCNConv(2 * self.c_in, self.c_out, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 5, 6]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, 1)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:
            return F.layer_norm(
                input, tuple(input.shape[1:]), self.weight, self.bias, self.eps
            )

    def extra_repr(self):
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
