'''
Modified from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False, residual=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj):
        support = torch.matmul(features, self.weight)
        output = torch.mm(adj, support)

        if self.residual:
            output = output + features
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_edge(nn.Module):
    def __init__(self, in_features, out_features, bias=False, residual=False):
        super(GraphConvolution_edge, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, mat_inci):
        mat_inci = mat_inci - 2 * torch.eye(mat_inci.size(-1), device=mat_inci.device)
        support = torch.matmul(features, self.weight)
        output = torch.bmm(mat_inci, support)

        if self.residual:
            output = output + features
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


## Attention Related
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features: int, out_features: int, n_heads: int,
#                  is_concat: bool = True,
#                  dropout: float = 0.6,
#                  leaky_relu_negative_slope: float = 0.2):
#         super(GraphAttentionLayer, self).__init__()
#
#         self.is_concat = is_concat
#         self.n_heads = n_heads
#
#         if is_concat:
#             assert out_features % n_heads == 0
#             self.n_hidden = out_features // n_heads
#         else:
#             self.n_hidden = out_features
#
#         self.linear = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
#         self.attn = nn.Linear(in_features=self.n_hidden * 2, out_features=1, bias=False) # Linear layer to compute the attention score e_ij
#         self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
#         '''
#             h is the input node embeddings of shape [n_nodes, in_features]
#             adj_mat is the adjacency matrix of shape [n_nodes, n_nodes, n_heads], use shape [n_nodes, n_nodes, 1] since the adj_mat is the same for each head.
#         '''
#         n_nodes = h.shape[0] # number of nodes
#         print(h.size())
#         print(self.linear(h).size())
#         g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden) # the initial transformation, g = Wh
#
#         # calculate [g_i || g_j]
#         g_repeat = g.repeat(n_nodes, 1, 1) # {g1, g2, ..., gn, g1, g2, ... gn, ...}
#         g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0) # {g1, g1, ..., g1, g2, g2, ..., g2, ...}
#         g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1) # {g1||g1, g1||g2, ..., g1||gn, ...}
#         g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden) # g_concat[i, j] is g_i||g_j
#
#         e = self.activation(self.attn(g_concat)) # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [n_nodes, n_nodes, n_heads, 1]
#         e = e.squeeze(dim=-1)
#
#         # adj_mat should have shape [n_nodes, n_nodes, n_heads] or [n_nodes, n_nodes, 1]
#         assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
#         assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
#         assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
#
#         e = e.mask_fill(adj_mat == 0, float('-inf'))
#         a = self.softmax(e)
#         a = self.dropout(a)
#         attn_res = torch.einsum('ijh, jhf->ihf', a, g) # h_i = sum(a_ij * g_j)
#
#         if self.is_concat:
#             return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
#         else:
#             return attn_res.mean(dim=-1)
#
#
# # 计算每一条边针对每一个顶点的Attention
# class GraphAttentionLayer_edge(nn.Module):
#     def __init__(self, in_features: int, out_features: int, n_heads: int,
#                  is_concat: bool = True,
#                  dropout: float = 0.6,
#                  leaky_relu_negative_slope: float = 0.2):
#         super(GraphAttentionLayer_edge, self).__init__()
#
#         self.is_concat = is_concat
#         self.n_heads = n_heads
#
#         if is_concat:
#             assert out_features % n_heads == 0
#             self.n_hidden = out_features // n_heads
#         else:
#             self.n_hidden = out_features
#
#         self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
#         self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
#
#         self.attn = nn.Linear(in_features=self.n_hidden * 2, out_features=1, bias=False)  # Linear layer to compute the attention score e_ij
#         self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, vertex: torch.Tensor, edge: torch.Tensor, inci_mat: torch.Tensor):
#         '''
#             vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
#             edge is the input edges embeddings of shape [batch, n_edges, in_features]
#             inci_mat is the incidence matrix of the graph, need to be transposed
#         '''
#         n_nodes = vertex.shape[1]  # number of nodes
#         n_edges = edge.shape[1] # number of edges
#         g_vertex = self.linear_vert(vertex).view(n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh
#         g_edge = self.linear_edge(edge).view(n_edges, self.n_heads, self.n_hidden)
#
#         # calculate similarity(edge_i, vertex_j), is the QK in the attention formula
#         g_head = []
#
#         for i in range(self.n_heads):
#             g_head.append(torch.bmm(g_edge[:, :, i, :], g_vertex[:, :, i, :].permute(0, 2, 1)).unsqueeze(dim=-2))
#
#         g_concat = torch.cat(g_head, dim=-2)
#         e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [n_nodes, n_nodes, n_heads, 1]
#         e = e.squeeze(dim=-1)
#
#         e = e.mask_fill(inci_mat == 0, float('-inf'))
#         a = self.softmax(e)
#         a = self.dropout(a)
#         attn_res = torch.einsum('ijh, jhf->ihf', a, g_vertex)  # h_i = sum(a_ij * g_j)
#
#         if self.is_concat:
#             return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
#         else:
#             return attn_res.mean(dim=-1)

# 计算每一条边针对每一个顶点的Attention
class GraphAttentionLayer_edge(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(GraphAttentionLayer_edge, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(in_features=n_heads, out_features=n_heads, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor, edge: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
            edge is the input edges embeddings of shape [batch, n_edges, in_features]
            inci_mat is the incidence matrix of the graph, need to be transposed
        '''
        batch_size = vertex.shape[0]
        n_nodes = vertex.shape[1]  # number of nodes
        n_edges = edge.shape[1]  # number of edges


        g_vertex = self.linear_vert(vertex).view(batch_size, n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]
        g_edge = self.linear_edge(edge).view(batch_size, n_edges, self.n_heads, self.n_hidden)  # [batch_size, n_edge, n_head, n_hidden]

        # calculate similarity(edge_i, vertex_j), is the QK in the attention formula
        g_head = []
        for i in range(self.n_heads):
            g_head.append(torch.bmm(g_edge[:, :, i, :], g_vertex[:, :, i, :].permute(0, 2, 1)).unsqueeze(dim=-1))  # [batch_size, n_edge, n_node, 1]
        g_concat = torch.cat(g_head, dim=-1)  # [batch_size, n_edge, n_node, n_head]
        e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [batch_size, n_edge, n_node, n_head]
        e = e.squeeze(dim=-1).sum(dim=-1)

        return e


class GraphAttentionLayer_Cora_edge(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(GraphAttentionLayer_Cora_edge, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.linear_edge = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(in_features=n_heads, out_features=n_heads, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor, edge: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
            edge is the input edges embeddings of shape [batch, n_edges, in_features]
            inci_mat is the incidence matrix of the graph, need to be transposed
        '''
        n_nodes = vertex.shape[0]  # number of nodes
        n_edges = edge.shape[0]  # number of edges

        g_vertex = self.linear_vert(vertex).view(n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]
        g_edge = self.linear_edge(edge).view(n_edges, self.n_heads, self.n_hidden)  # [batch_size, n_edge, n_head, n_hidden]

        # calculate similarity(edge_i, vertex_j), is the QK in the attention formula
        g_head = []

        for i in range(self.n_heads):
            g_head.append(torch.mm(g_edge[:, i, :], g_vertex[:, i, :].t()).unsqueeze(dim=-1))  # [batch_size, n_edge, n_node, 1]

        g_concat = torch.cat(g_head, dim=-1)  # [batch_size, n_edge, n_node, n_head]

        # g_concat = torch.mm(g_edge, g_vertex.t())

        e = self.activation(self.attn(g_concat))  # e_ij = LeakyReLU(a^T [g_i||g_j]), the shape of e is [batch_size, n_edge, n_node, n_head]
        e = e.squeeze(dim=-1).sum(dim=-1)

        return e


# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features: int, out_features: int, n_heads: int,
#                  is_concat: bool = True,
#                  dropout: float = 0.6,
#                  leaky_relu_negative_slope: float = 0.2):
#         super(GraphAttentionLayer, self).__init__()
#
#         self.is_concat = is_concat
#         self.n_heads = n_heads
#
#         if is_concat:
#             assert out_features % n_heads == 0
#             self.n_hidden = out_features // n_heads
#         else:
#             self.n_hidden = out_features
#
#         self.linear = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
#         self.attn = nn.Linear(in_features=self.n_hidden * 2, out_features=1, bias=False)
#         self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
#         self.softmax = nn.Softmax(dim=-2)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
#         batch_size = h.shape[0]
#         n_nodes = h.shape[1]
#
#         g = self.linear(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
#         g_repeat = g.repeat(1, n_nodes, 1, 1)
#         g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)
#         g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
#         g_concat = g_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden * 2)
#         e = self.activation(self.attn(g_concat))
#         e = e.squeeze(dim=-1)
#         zero_vec = -9e15 * torch.ones_like(e)
#
#         a = torch.where(adj_mat.unsqueeze(dim=-1).repeat_interleave(self.n_heads, dim=-1) > 0, e, zero_vec)
#         a = self.softmax(a)
#         a = self.dropout(a)
#         attn_res = torch.einsum('kijh, kjhf->kihf', a, g)
#
#         if self.is_concat:
#             return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
#         else:
#             return attn_res.mean(dim=-2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(in_features=self.n_hidden * 2, out_features=1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        # batch_size = h.shape[0]
        n_nodes = h.shape[0]

        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, self.n_hidden * 2)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(dim=-1)

        adj_mat = adj_mat.unsqueeze(dim=-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)

        # zero_vec = -9e15 * torch.ones_like(e)

        # a = torch.where(adj_mat.unsqueeze(dim=-1).repeat_interleave(self.n_heads, dim=-1) > 0, e, zero_vec)
        # a = self.softmax(a)
        # a = self.dropout(a)
        attn_res = torch.einsum('ijh, jhf->ihf', a, g)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=-2)


class _gat_layer_batch(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        super(_gat_layer_batch, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_vert = nn.Linear(in_features=in_features, out_features=self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(in_features=self.n_hidden * 2, out_features=1, bias=False)  # Linear layer to compute the attention score e_ij
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vertex: torch.Tensor):
        '''
            vertex is the input nodes embeddings of shape [batch, n_nodes, in_features]
        '''
        batch_size = vertex.shape[0]
        n_nodes = vertex.shape[1]  # number of nodes

        g_vertex = self.linear_vert(vertex).view(batch_size, n_nodes, self.n_heads, self.n_hidden)  # the initial transformation, g = Wh, [batch_size, n_node, n_head, n_hidden]

        g_rep = g_vertex.repeat(1, n_nodes, 1, 1)
        g_rep_inter = g_vertex.repeat_interleave(n_nodes, dim=1)

        g_cat = torch.cat([g_rep_inter, g_rep], dim=-1)
        g_cat = g_cat.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden * 2)

        e = self.activation(self.attn(g_cat))
        e = e.squeeze(dim=-1)

        a = self.softmax(e)
        a = self.dropout(a)

        attr_res = torch.einsum('bijh,bjhf->bihf', a, g_vertex)

        if self.is_concat:
            return attr_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attr_res.mean(dim=-1)


class _attention_layer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(_attention_layer, self).__init__()

        self.w_omega = nn.Parameter(torch.Tensor(in_features, in_features))
        self.u_omega = nn.Parameter(torch.Tensor(in_features, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, vertex):
        # w_omega = self.w_omega.unsqueeze(dim=0)
        # w_omega = w_omega.repeat(vertex.size(0), 1, 1)
        # u_omega = self.u_omega.unsqueeze(dim=0)
        # u_omega = u_omega.repeat(vertex.size(0), 1, 1)

        u = torch.tanh(torch.matmul(vertex, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=-1)

        scored_vertex = vertex * att_score

        feat = torch.sum(scored_vertex, dim=1)

        return feat


class GraphAttentionLayer_Cora(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    # in_features=in_features, out_features=n_hidden, n_heads=n_heads, is_concat=True, dropout=dropout
    def __init__(self, in_features: int, out_features: int, is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):

                 # in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_Cora, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.concat = is_concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)


    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Transformer Related
# class PrepareForMultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, n_heads: int, d_k: int, bias: bool):
#         super(PrepareForMultiHeadAttention, self).__init__()
#         self.linear = nn.Linear(d_model, n_heads * d_k, bias=bias)
#         self.n_heads = n_heads
#         self.d_k = d_k
#
#     def forward(self, x: torch.Tensor):
#         head_shape = x.shape[:-1]
#         x = self.linear(x)
#         x = x.view(*head_shape, self.n_heads, self.d_k)
#
#         return x
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads: int, d_model: int, dropout: float=0.1, bias=True):
#         super(MultiHeadAttention, self).__init__()
#         self.d_k = d_model // n_heads
#         self.n_heads = n_heads
#         self.query = PrepareForMultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=self.d_k, bias=bias)
#         self.key = PrepareForMultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=self.d_k, bias=bias)
#         self.value = PrepareForMultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=self.d_k, bias=bias)
#         self.softmax = nn.Softmax(dim=1)
#         self.output = nn.Linear(in_features=d_model, out_features=d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.scale = 1 / math.sqrt(self.d_k)
#         self.attn = None
#
#     def get_scores(self, query: torch.Tensor, key: torch.Tensor):
#         return torch.einsum('ibhd, jbhd->ijbh', query, key)
#
#     def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
#         assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
#         assert mask.shape[1] == key_shape[0]
#         assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
#
#         mask = mask.unsqueeze(dim=-1)
#
#         return mask

# def forward(self, *,
#             query: torch.Tensor,
#             key: torch.Tensor,
#             value: torch.Tensor,
#             mask: Optional[torch.Tensor] = None):
#     seq_len, batch_size, _
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


class PositionalEncoding_Cora(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding_Cora, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x.to_dense() + pe
        x = x.to_sparse()
        # x = pe + x
        # x = self.dropout(x)
        return x


def get_positional_encoding(d_model: int, max_len: int = 5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    tow_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(tow_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(dim=1).requires_grad_(False)

    return encodings
#
# def _test_positional_encoding():
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     plt.figure(figsize=(15, 5))
#     pe = get_positional_encoding(20, 100)
#     plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
#     plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
#     plt.title('Positional Encoding')
#     plt.show()
#
# if __name__ == '__main__':
#     _test_positional_encoding()
