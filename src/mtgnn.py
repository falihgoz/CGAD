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

# from src.layers import *
import torch
import torch.nn as nn
from src.layers import dilated_inception, GCN, LayerNorm
import torch.nn.functional as F


class gtnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        edge_index=None,
        edge_weight=None,
        seq_in_len=15,
        gcn_true=True,
    ):
        super(gtnet, self).__init__()
        self.seq_length = seq_in_len
        self.conv_channels = 16
        self.residual_channels = 16
        self.skip_channels = 32
        self.end_channels = 64
        self.layers = 3
        kernel_size = 6
        self.dropout = 0.0

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=1, out_channels=self.residual_channels, kernel_size=(1, 1)
        )

        self.gcn_true = gcn_true
        self.num_nodes = num_nodes
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device, dtype=torch.float)

        self.receptive_field = self.layers * (kernel_size - 1) + 1

        for i in range(1, self.layers + 1):
            rf_size_i = 1 + i * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(self.residual_channels, self.conv_channels)
            )
            self.gate_convs.append(
                dilated_inception(self.residual_channels, self.conv_channels)
            )

            self.residual_convs.append(
                nn.Conv2d(
                    in_channels=self.conv_channels,
                    out_channels=self.residual_channels,
                    kernel_size=(1, 1),
                )
            )

            if self.seq_length > self.receptive_field:
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=self.conv_channels,
                        out_channels=self.skip_channels,
                        kernel_size=(1, self.seq_length - rf_size_i + 1),
                    )
                )
            else:
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=self.conv_channels,
                        out_channels=self.skip_channels,
                        kernel_size=(1, self.receptive_field - rf_size_i + 1),
                    )
                )

            if self.gcn_true:
                self.gcn.append(
                    GCN(self.conv_channels, self.residual_channels, self.dropout)
                )

            if self.seq_length > self.receptive_field:
                self.norm.append(
                    LayerNorm(
                        (
                            self.residual_channels,
                            num_nodes,
                            self.seq_length - rf_size_i + 1,
                        ),
                        elementwise_affine=True,
                    )
                )
            else:
                self.norm.append(
                    LayerNorm(
                        (
                            self.residual_channels,
                            num_nodes,
                            self.receptive_field - rf_size_i + 1,
                        ),
                        elementwise_affine=True,
                    )
                )

        self.layers = self.layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=self.skip_channels,
            out_channels=self.end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.end_channels, out_channels=1, kernel_size=(1, 1), bias=True
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=1,
                out_channels=self.skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=self.residual_channels,
                out_channels=self.skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=1,
                out_channels=self.skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=self.residual_channels,
                out_channels=self.skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input):
        seq_len = input.size(3)
        assert (
            seq_len == self.seq_length
        ), "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_true:
                x = torch.permute(x, (0, 3, 2, 1))
                x = self.gcn[i](x, self.edge_index, self.edge_weight)
                x = torch.permute(x, (0, 3, 2, 1))
            else:
                x = self.residual_convs[i](x)

            idx = None
            x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
