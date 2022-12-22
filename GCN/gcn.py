'''
modified from https://github.com/tkipf/pygcn/blob/master/pygcn/models.py

The MIT License

Copyright (c) 2017 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *
from torch_geometric.nn import GCNConv

torch.cuda.manual_seed(1)
device = torch.device("cuda:0")
# device = torch.device("cpu")

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = F.softmax(x, dim=1)

        return x


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden, num_classes):
        super(GCN, self).__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            in_channels = in_features if i == 0 else hidden
            self.layers.append(GraphConvolution(in_channels, hidden, residual=False))
        self.last_layer = GraphConvolution(hidden, num_classes)

    def forward(self, x, adj):
        for l in self.layers:
            x = F.relu(l(x, adj))
        x = self.last_layer(x, adj)
        # x = F.relu(x)
        return x


class HGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(HGCN, self).__init__()
        self.weight_1 = self.glorot_init(input_dim=in_features, output_dim=out_features)
        # self.weight_2 = self.glorot_init(input_dim=hidden, output_dim=out_features)
        self.activation = F.relu

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, adj, res):
        x = inputs
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight_1)
        x = x + res
        x = self.activation(x)

        return x


class GCN_edge(nn.Module):
    def __init__(self, n_layers, in_features, hidden, num_classes):
        super(GCN_edge, self).__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            in_channels = in_features if i == 0 else hidden
            self.layers.append(GraphConvolution_edge(in_channels, hidden))
        self.last_layer = GraphConvolution_edge(hidden, num_classes)

    def forward(self, x, adj):
        for l in self.layers:
            x = F.relu(l(x, adj))
        x = self.last_layer(x, adj)
        # x = F.relu(x)
        return x


class GCN_D(nn.Module):
    def __init__(self, in_features, hidden, num_classes, activation=lambda x: x):
        super(GCN_D, self).__init__()

        self.layer_1 = GraphConvSparse(input_dim=in_features, output_dim=hidden)
        self.layer_2 = GraphConvSparse(input_dim=hidden, output_dim=num_classes, activation=activation)

    def forward(self, x, adj):
        x = self.layer_1(x, adj)
        x = self.layer_2(x, adj.permute(0, 2, 1))
        return x


# class GCN_DAD(nn.Module):
#     def __init__(self, in_features, hidden, num_classes):
#         super(GCN_DAD, self).__init__()
#
#         self.layer_1 = GraphConvSparse(input_dim=in_features, output_dim=hidden)
#         self.layer_2 = GraphConvSparse(input_dim=hidden, output_dim=num_classes, activation=lambda x: x)
#
#     def forward(self, x, adj):
#         x = self.layer_1(x, adj)
#         x = self.layer_2(x, adj.permute(0, 2, 1))
#         return x


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.bmm(x, torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=x.size(0), dim=0))
        x = torch.bmm(adj, x)
        outputs = self.activation(x)
        return outputs


class GraphConvSparse_Edge(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_Edge, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, adj):
        x = inputs
        # x = torch.bmm(x, torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=x.size(0), dim=0))
        x = torch.bmm(adj, x)
        # outputs = self.activation(x)
        return x  # outputs


class GraphConvSparse_Cora_Edge(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_Cora_Edge, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        # output_dim = 1
        # init_range = np.sqrt(1.0 / (input_dim + output_dim))
        # initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        initial = torch.zeros(input_dim, output_dim) + 1e-3

        return nn.Parameter(initial)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight) # torch.repeat_interleave(self.weight, repeats=x.size(1), dim=1))
        x = torch.mm(adj, x)

        outputs = self.activation(x)
        return outputs


class GraphConvSparse_WikiCS_Edge(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_WikiCS_Edge, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        output_dim = 1
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial).to(device)

    def forward(self, inputs, adj):
        x = inputs

        # x = torch.bmm(x, torch.repeat_interleave(self.weight, repeats=x.size(1), dim=1))
        x = torch.bmm(adj, x)

        # outputs = self.activation(x)
        return x# outputs


class GraphConvSparse_Edge_Inci(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_Edge_Inci, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial)

    def forward(self, inputs, inci_mat):
        adj = torch.bmm(inci_mat.permute(0, 2, 1), inci_mat)
        x = inputs

        x = torch.bmm(x, torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=x.size(0), dim=0))
        x = torch.bmm(adj, x)
        outputs = self.activation(x)
        return outputs


class GraphInceptionConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphInceptionConv, self).__init__()

        self.layer_1 = GraphConvSparse_Edge_Inci(input_dim=input_dim, output_dim=hidden_dim * 0 + out_dim)
        # self.layer_2 = GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)

    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)
        # x_2 = self.layer_2(x_1, adj)

        # out = torch.cat([x_1, x_2], dim=-1)
        return x_1 # out


class GraphInceptionConv_mean(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphInceptionConv_mean, self).__init__()

        self.layer_1 = GraphConvSparse_Edge_Inci(input_dim=input_dim, output_dim=hidden_dim * 0 + out_dim)
        # self.layer_2 = GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)

    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)
        # x_2 = self.layer_2(x_1, adj)

        # out = torch.cat([x_1, x_2], dim=-1)
        return x_1 # out


class GraphConvSparse_Cora_Edge_Inci(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_Cora_Edge_Inci, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        # init_range = np.sqrt(0.1 / (input_dim + output_dim))
        # initial = torch.rand(input_dim, output_dim) * 0.1 * init_range - init_range

        initial = torch.zeros(input_dim, output_dim) + 1e-6
        return nn.Parameter(initial)

    def forward(self, inputs, inci_mat):
        adj = torch.mm(inci_mat.t(), inci_mat)
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)

        outputs = self.activation(x)
        return outputs


class GraphConvSparse_Cora_Node_Inci(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse_Cora_Node_Inci, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        # init_range = np.sqrt(0.1 / (input_dim + output_dim))
        # initial = torch.rand(input_dim, output_dim) * 0.1 * init_range - init_range
        initial = torch.zeros(input_dim, output_dim) + 1e-6
        return nn.Parameter(initial)

    def forward(self, inputs, adj):#inci_mat):
        # adj = torch.mm(inci_mat.t(), inci_mat)

        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class GraphInceptionConv_Cora(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphInceptionConv_Cora, self).__init__()


        self.layer_1 = GraphConvSparse_Cora_Edge_Inci(input_dim=input_dim, output_dim= hidden_dim * 0 + out_dim)
        # self.layer_2 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=hidden_dim + out_dim)
        # self.layer_3 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)
        # self.merge = nn.Linear(in_features=hidden_dim * 1 + out_dim, out_features=out_dim)


    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)
        # x_2 = self.layer_2(x_1, adj)
        # x_3 = self.layer_3(x_2, adj)
        # out = torch.cat([x_1, x_2], dim=-1)
        # out = torch.cat([out, x_3], dim=-1)

        # out = self.merge(torch.cat([out, x_3], dim=-1))
        return x_1


class GraphInceptionConv_Cora_Node(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GraphInceptionConv_Cora_Node, self).__init__()


        self.layer_1 = GraphConvSparse_Cora_Node_Inci(input_dim=input_dim, output_dim= hidden_dim) # * 2 + out_dim)
        self.layer_2 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=hidden_dim) # + out_dim)
        self.layer_3 = GraphConvSparse_Cora_Edge_Inci(input_dim=hidden_dim, output_dim=out_dim, activation=lambda x: x)
        # self.merge = nn.Linear(in_features=hidden_dim * 1 + out_dim, out_features=out_dim)


    def forward(self, x, adj):
        x_1 = self.layer_1(x, adj)

        x_2 = self.layer_2(x_1, adj)
        x_3 = self.layer_3(x_2, adj)
        out = torch.cat([x_1, x_2], dim=-1)
        out = torch.cat([out, x_3], dim=-1)

        # out = self.merge(torch.cat([out, x_3], dim=-1))
        return out


class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, dropout: float):
        super(GAT, self).__init__()
        self.attn_layer_1 = GraphAttentionLayer(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()
        self.attn_layer_2 = GraphAttentionLayer(in_features=n_hidden, out_features=out_features, n_heads=1, is_concat=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = self.attn_layer_1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.attn_layer_2(x, adj_mat)

        return x


class GAT_PE(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, is_concat: bool, dropout: float):
        super(GAT_PE, self).__init__()
        self.attn_layer = GraphAttentionLayer(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=is_concat, dropout=dropout)
        self.activation = nn.ELU()
        self.pos_encoder = PositionalEncoding(d_model=in_features, dropout=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.pos_encoder(x)
        x = self.attn_layer(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)

        return x

# in_features=1434, n_hidden=hidden_dim * 8, out_features=hidden_dim, n_heads=8, dropout=0.1
class GAT_Cora(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, is_concat: bool, dropout: float):
        """Dense version of GAT."""
        super(GAT_Cora, self).__init__()
        self.dropout = dropout

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.attentions = [GraphAttentionLayer_Cora(in_features=in_features, out_features=self.n_hidden, dropout=dropout, is_concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.pos_encoder(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GAT_DE(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, dropout: float):
        super(GAT_DE, self).__init__()
        self.attn_layer = GraphAttentionLayer_edge(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, vert: torch.Tensor, edge: torch.Tensor):
        # x = self.dropout(x)
        x = self.attn_layer(vert, edge)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class GAT_Cora_DE(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, out_features: int, n_heads: int, dropout: float):
        super(GAT_Cora_DE, self).__init__()
        self.attn_layer = GraphAttentionLayer_Cora_edge(in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout)
        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, vert: torch.Tensor, edge: torch.Tensor):
        # x = self.dropout(x)
        x = self.attn_layer(vert, edge)
        x = self.activation(x)
        x = self.dropout(x)

        return x