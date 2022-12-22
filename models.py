import math
import random
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb
from GCN.gcn import *
from GCN.layers import GraphConvolution
import networkx as nx
from sklearn.metrics import f1_score
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv, APPNP
from torch_scatter import scatter_add
import scipy
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter, Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.utils.num_nodes import maybe_num_nodes
from GCN.layers import _gat_layer_batch, _attention_layer

torch.cuda.manual_seed(1)
device = torch.device("cuda:0")

class DVAE_INCI_WO_GNN(nn.Module):
    def __init__(self, max_n, max_edge_n, n_vertex_type, START_TYPE, END_TYPE, hidden_dim=501, z_dim=56):
        super(DVAE_INCI_WO_GNN, self).__init__()
        self.max_n_vertex = max_n  # maximum number of vertices
        self.max_n_edge = max_edge_n  # max number of edges
        self.n_vertex_type = n_vertex_type  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hidden_dim  # hidden state size of each vertex
        self.nz = z_dim  # size of latent representation z
        self.gs = hidden_dim  # size of graph state
        self.device = None

        # 0. encoding part
        # 0.1 topology encoding
        self.embedding_ops = nn.Embedding(n_vertex_type, hidden_dim * 20)

        self.mu_vert = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 20, out_features=z_dim + hidden_dim * 0),
            nn.ReLU(),
        )

        self.logvar_vert = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 20, out_features=z_dim + hidden_dim * 0),
            nn.ReLU(),
        )

        # 0.2 orientation encoding
        self.gnn_edge_T = GraphConvSparse_Edge(input_dim=hidden_dim * 20, output_dim=hidden_dim * 20)
        self.mu_edge_T = GraphInceptionConv(input_dim=hidden_dim * 20, hidden_dim=hidden_dim, out_dim=z_dim) # GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=z_dim)  # nn.Linear(in_features=hidden_dim, out_features=z_dim) # GCN_D(in_features=250, hidden=128, num_classes=z_dim)
        self.logvar_edge_T = GraphInceptionConv(input_dim=hidden_dim * 20, hidden_dim=hidden_dim, out_dim=z_dim) #GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=z_dim)  # nn.Linear(in_features=hidden_dim, out_features=z_dim) # GCN_D(in_features=250, hidden=128, num_classes=z_dim)

        self.gnn_edge_H = GraphConvSparse_Edge(input_dim=hidden_dim * 20, output_dim=hidden_dim * 20)
        self.mu_edge_H = GraphInceptionConv(input_dim=hidden_dim * 20, hidden_dim=hidden_dim, out_dim=z_dim) #GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=z_dim)  # nn.Linear(in_features=hidden_dim, out_features=z_dim)
        self.logvar_edge_H = GraphInceptionConv(input_dim=hidden_dim * 20, hidden_dim=hidden_dim, out_dim=z_dim) #GraphConvSparse_Edge_Inci(input_dim=hidden_dim, output_dim=z_dim)  # nn.Linear(in_features=hidden_dim, out_features=z_dim)

        # 1. decoding part
        self.gnn_inci_T = GAT_DE(in_features=z_dim + hidden_dim * 0, n_hidden=hidden_dim, out_features=hidden_dim, n_heads=8, dropout=0.1)
        self.gnn_inci_H = GAT_DE(in_features=z_dim + hidden_dim * 0, n_hidden=hidden_dim, out_features=hidden_dim, n_heads=8, dropout=0.1)

        self.add_vertex = nn.Sequential(
            nn.Linear(self.nz + hidden_dim * 0, self.hs),
            nn.ReLU(),
            nn.Linear(self.hs, self.n_vertex_type),
        )

        self.vertice_up = nn.Sequential(
            nn.Linear(z_dim, self.hs // 2),
            nn.ReLU(),
            nn.Linear(self.hs // 2, hidden_dim)
        )

        self.edge_up_T = nn.Sequential(
            nn.Linear(z_dim, self.hs // 2),
            nn.ReLU(),
            nn.Linear(self.hs // 2, hidden_dim)
        )

        self.edge_up_H = nn.Sequential(
            nn.Linear(z_dim, self.hs // 2),
            nn.ReLU(),
            nn.Linear(self.hs // 2, hidden_dim)
        )

        # 2. Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * max_n * z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.weight = self.glorot_init(max_edge_n, hidden_dim)


        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(dim=1)
        self.vertex_criterion = nn.CrossEntropyLoss()
        self.adj_criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='sum')

        self.layerNorm_vert = nn.LayerNorm([self.max_n_vertex, hidden_dim * 20])
        self.layerNorm_edge_T = nn.LayerNorm([self.max_n_edge, hidden_dim * 20])
        self.layerNorm_edge_H = nn.LayerNorm([self.max_n_edge, hidden_dim * 20])
        self.layerNorm_vert_de = nn.LayerNorm([self.max_n_vertex, hidden_dim * 0 + z_dim])
        self.layerNorm_edge_T_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 0 + z_dim])
        self.layerNorm_edge_H_de = nn.LayerNorm([self.max_n_edge, hidden_dim * 0 + z_dim])

        self.layerNorm_mean_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 0])
        self.layerNorm_mean_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 0])
        self.layerNorm_mean_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 0])
        self.layerNorm_logvar_vert = nn.LayerNorm([max_n, z_dim + hidden_dim * 0])
        self.layerNorm_logvar_edge_T = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 0])
        self.layerNorm_logvar_edge_H = nn.LayerNorm([max_edge_n, z_dim + hidden_dim * 0])


    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(1.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range

        return nn.Parameter(initial)

    def encode(self, arch):
        _, _, _, _, ops, inci_mat_T, inci_mat_H, _, _, _, _, _, _, _ = arch

        if torch.is_tensor(inci_mat_T):
            inci_mat_T = inci_mat_T.to(device)
            ops = ops.to(device)
            inci_mat_H = inci_mat_H.to(device)

        if not torch.is_tensor(inci_mat_T):
            inci_mat_T = torch.cat(inci_mat_T, dim=0)
            inci_mat_H = torch.cat(inci_mat_H, dim=0)
            ops = torch.cat(ops, dim=0)

        hidden_vert = self.embedding_ops(ops)

        edges_attn_T = self.gnn_edge_T(hidden_vert, inci_mat_T.permute(0, 2, 1))
        edges_attn_H = self.gnn_edge_H(hidden_vert, inci_mat_H.permute(0, 2, 1))

        mean_vert = self.mu_vert(hidden_vert)
        logvar_vert = self.logvar_vert(hidden_vert)

        # edge attention
        mean_edge_T = self.mu_edge_T(edges_attn_T, inci_mat_T)
        logvar_edge_T = self.logvar_edge_T(edges_attn_T, inci_mat_T)

        mean_edge_H = self.mu_edge_H(edges_attn_H, inci_mat_H)
        logvar_edge_H = self.logvar_edge_H(edges_attn_H, inci_mat_H)

        mean_vert = self.layerNorm_mean_vert(mean_vert)
        mean_edge_T = self.layerNorm_mean_edge_T(mean_edge_T)
        mean_edge_H = self.layerNorm_mean_edge_H(mean_edge_H)
        logvar_vert = self.layerNorm_logvar_vert(logvar_vert)
        logvar_edge_T = self.layerNorm_logvar_edge_T(logvar_edge_T)
        logvar_edge_H = self.layerNorm_logvar_edge_H(logvar_edge_H)

        z_vertex = self._GaussianNoise(mean_vert.size()) * torch.exp(0.5 * logvar_vert) + mean_vert
        z_edge_T = self._GaussianNoise(mean_edge_T.size()) * torch.exp(0.5 * logvar_edge_T) + mean_edge_T
        z_edge_H = self._GaussianNoise(mean_edge_H.size()) * torch.exp(0.5 * logvar_edge_H) + mean_edge_H

        return (mean_vert, mean_edge_T, mean_edge_H), (logvar_vert, logvar_edge_T, logvar_edge_H), (z_vertex, z_edge_T, z_edge_H)

    def _GaussianNoise(self, size):
        gaussian_noise = torch.rand(size).to(device)
        return gaussian_noise

    def _reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _is_valid_DAG(self, g, START_TYPE=0, END_TYPE=1):
        res = g.is_dag()
        n_start, n_end = 0, 0
        for v in g.vs:
            if v['type'] == START_TYPE:
                n_start += 1
            elif v['type'] == END_TYPE:
                n_end += 1
            if v.indegree() == 0 and v['type'] != START_TYPE:
                return False
            if v.outdegree() == 0 and v['type'] != END_TYPE:
                return False
        return res and n_start == 1 and n_end == 1

    def _to_graph(self, ops, inci):
        ops = ops.cpu().tolist()

        ops[0] = self.START_TYPE
        ops[-1] = self.END_TYPE

        ops_ori = ops
        trunc = []
        for i, item in enumerate(ops):
            if item == self.START_TYPE:
                trunc.append(i)

        ops = ops[trunc[-1]:]

        for i, item in enumerate(ops):
            if item == self.END_TYPE:
                ops = ops[:i + 1]

        g = igraph.Graph(directed=True)

        g.add_vertices(len(ops))
        edges = []

        for i in range(len(ops)):
            g.vs[i]['type'] = ops[i]

        for i in range(inci.size(0)):
            tail = -1
            head = -1
            for j in range(len(ops)):
                if inci[i][j] == 1:
                    head = j
                if inci[i][j] == -1:
                    tail = j
            if tail != -1 and head != -1 and tail < head:
                edges.append((tail, head))

        edges = list(set(edges))
        g.add_edges(edges)

        single_vertices = set([v.index for v in g.vs.select(_indegree_eq=0)
                               if v.index != 0])

        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0)
                            if v.index != g.vcount() - 1])

        for v in end_vertices:
            g.add_edge(v, g.vcount() - 1)

        for v in single_vertices:
            g.add_edge(0, v)

        if not self._is_valid_DAG(g, self.START_TYPE, self.END_TYPE):
            print('*' * 20)
            print(ops_ori)
            print(g)
            for v in range(g.vcount()):
                print(g.vs[v])
            print('#' * 20)

        # print('decode:', ops_ori, inci)

        return g

    def calculate_accuracy(self, z, stochastic=True):
        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.2] = 0.
        inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.2] = 0.

        inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
        inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
        inci_pred = inci_T + inci_H

        v_type_pred = torch.max(vertex_pred, dim=-1)[1].view(z[0].size(0), -1)

        return v_type_pred, inci_pred

    def _decode(self, z):
        z_vert = z[0]
        z_edge_T = z[1]
        z_edge_H = z[2]

        z_vert = self.layerNorm_vert_de(z_vert)
        z_edge_T = self.layerNorm_edge_T_de(z_edge_T)
        z_edge_H = self.layerNorm_edge_H_de(z_edge_H)

        inci_pred_T = self.gnn_inci_T(z_vert, z_edge_T)
        inci_pred_H = self.gnn_inci_H(z_vert, z_edge_H)

        vertex_pred = self.add_vertex(z_vert)
        vertex_pred = F.softmax(vertex_pred, dim=-1)

        return vertex_pred, inci_pred_T, inci_pred_H

    def decode(self, z):
        G = []

        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T[inci_pred_T.max(dim=-1)[0] < 0.2] = 0.
        inci_pred_H[inci_pred_H.max(dim=-1)[0] < 0.2] = 0.

        inci_T = F.one_hot(torch.argmax(inci_pred_T, dim=-1, keepdim=False), num_classes=self.max_n_vertex) * -1
        inci_H = F.one_hot(torch.argmax(inci_pred_H, dim=-1, keepdim=False), num_classes=self.max_n_vertex)
        inci_pred = inci_T + inci_H

        v_type_pred = torch.max(vertex_pred, dim=-1)[1].view(z[0].size(0), -1)

        for (op, ic) in zip(v_type_pred, inci_pred):
            G.append(self._to_graph(op, ic))

        return G

    def loss(self, mean, logvar, z, G_true):
        acc, adj_mat, adj_N, nodes, ops, inci_mat_T, inci_mat_H, _, inci_lb_T, inci_lb_H, _, weight_T, weight_H, _ = G_true

        if torch.is_tensor(inci_mat_T):
            inci_lb_T = inci_lb_T.to(device)
            inci_lb_H = inci_lb_H.to(device)
            ops = ops.to(device)
            weight_T = weight_T.to(device)
            weight_H = weight_H.to(device)
            acc = acc.to(device)

        if not torch.is_tensor(inci_mat_T):
            inci_lb_T = torch.cat(inci_lb_T, dim=0)
            inci_lb_H = torch.cat(inci_lb_H, dim=0)
            ops = torch.cat(ops, dim=0)
            weight_T = torch.cat(weight_T, dim=0)
            weight_H = torch.cat(weight_H, dim=0)
            acc = torch.cat(acc, dim=0)

        mean_vert = mean[0]
        mean_edge_T = mean[1]
        mean_edge_H = mean[2]

        logvar_vert = logvar[0]
        logvar_edge_T = logvar[1]
        logvar_edge_H = logvar[2]

        vertex_pred, inci_pred_T, inci_pred_H = self._decode(z)

        inci_pred_T = self.sigmoid(inci_pred_T)
        inci_pred_H = self.sigmoid(inci_pred_H)

        edge_loss_T = F.binary_cross_entropy(inci_pred_T.permute(0, 2, 1).contiguous().view(-1), inci_lb_T.view(-1), reduction='mean', weight=weight_T)
        edge_loss_H = F.binary_cross_entropy(inci_pred_H.permute(0, 2, 1).contiguous().view(-1), inci_lb_H.view(-1), reduction='mean', weight=weight_H)

        acc_pred = self.pred_acc_2nd(mean_vert, inci_lb_T, inci_lb_H)
        acc_loss = self.mse_loss(acc_pred.squeeze(dim=-1), acc)
        vertex_loss = self.vertex_criterion(vertex_pred.view(-1, vertex_pred.size(-1)), ops.view(-1))

        mean_cat = [mean_vert, mean_edge_T, mean_edge_H]
        logvar_cat = [logvar_vert, logvar_edge_T, logvar_edge_H]
        mean_cat = torch.cat(mean_cat, dim=-2)
        logvar_cat = torch.cat(logvar_cat, dim=-2)

        kl_divergence = -0.5 * (1 + logvar_cat - mean_cat ** 2 - torch.exp(logvar_cat)).mean()

        loss = vertex_loss + edge_loss_H + edge_loss_T + kl_divergence + acc_loss
        return loss, vertex_loss, acc_loss, edge_loss_T, edge_loss_H, kl_divergence

    def pred_acc(self, inci_pred_T, inci_pred_H, inci_lb_T, inci_lb_H):
        acc_pred = (inci_pred_T + inci_pred_H) * (inci_lb_T.permute(0, 2, 1) * 1 + inci_lb_H.permute(0, 2, 1))
        weight = torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=inci_pred_T.size(0), dim=0)
        acc_pred = torch.bmm(acc_pred, weight)
        acc_pred = self.predictor(acc_pred.view(acc_pred.size(0), -1))

        return acc_pred

    def vec_x_mat(self, vec, mat):
        bt = vec.shape[0]
        h0 = vec.shape[1]
        h1 = vec.shape[2]
        h2 = mat.shape[1]
        w2 = mat.shape[2]

        vec = vec.view(bt, h0, 1, -1).expand([bt, h0, w2, h1])
        mat = mat.view(bt, h2, w2, 1).expand([bt, h2, w2, h1])


        return vec * mat

    # 不加入边的属性，完全是节点的属性
    def pred_acc_v(self, vert, edge, inci_T, inci_H):
        edge = torch.ones_like(edge).to(self.get_device())
        acc_pred = torch.bmm(edge, vert.permute(0, 2, 1))
        acc_pred = acc_pred * (inci_T.permute(0, 2, 1) + inci_H.permute(0, 2, 1))
        weight = torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=inci_T.size(0), dim=0)
        acc_pred = torch.bmm(acc_pred, weight)
        acc_pred = self.predictor(acc_pred.view(acc_pred.size(0), -1))

        return acc_pred

    # 现在考虑二阶近似问题，传递一次
    def pred_acc_2nd(self, vert, inci_T, inci_H):
        inci = inci_T + inci_H

        vmi = self.vec_x_mat(vert, inci)
        weight = torch.repeat_interleave(self.weight.unsqueeze(dim=0), repeats=inci_T.size(0), dim=0)
        mut = torch.einsum('bijk,bjd->bidk', vmi, weight)
        proped_vert = mut.mean(dim=2)

        # 2nd propagation
        vmi = self.vec_x_mat(proped_vert, inci)
        mut = torch.einsum('bijk,bjd->bidk', vmi, weight)
        acc_pred = self.predictor(mut.reshape(mut.size(0), -1))

        return acc_pred


    def link_predictor(self, Z, g_batch):
        _, adj_mat, edges_positive, edges_negative, ops, inci_mat_T, inci_mat_H, _, inci_lb_T, inci_lb_H, weight, weight_T, weight_H, _ = g_batch
        g_recon = self.decode(Z)

        pred = []
        label = []
        acc = 0

        n = len(g_recon)
        for (eg_p, eg_n, g) in zip(edges_positive, edges_negative, g_recon):
            edge_list = g.get_edgelist()
            eg_p = eg_p.squeeze().tolist()
            eg_p = (eg_p[0], eg_p[1])
            eg_n = eg_n.squeeze().tolist()
            eg_n = (eg_n[0], eg_n[1])

            if eg_p in edge_list:
                acc = acc + 1
                pred.append(1)
                label.append(1)

            else:
                pred.append(0)
                label.append(1)
            if eg_n not in edge_list:
                acc = acc + 1
                pred.append(0)
                label.append(0)
            else:
                pred.append(1)
                label.append(0)

        acc = float(acc / (n * 2))
        f1 = f1_score(label, pred)

        return acc, f1

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode_decode(self, G):
        mean, logvar, sampled_z = self.encode(G)
        return self.decode(sampled_z)

    def forward(self, G):
        mean, logvar, sampled_z = self.encode(G)
        loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = self.loss(mean, logvar, sampled_z, G)
        return loss

    def generate_sample(self, n):
        z_vertex = torch.randn(n, self.max_n_vertex, self.nz + self.hs * 0).to(self.get_device())
        z_edge_T = torch.randn(n, self.max_n_edge, self.nz + self.hs * 0).to(self.get_device())
        z_edge_H = torch.randn(n, self.max_n_edge, self.nz + self.hs * 0).to(self.get_device())
        sample = (z_vertex, z_edge_T, z_edge_H)
        G = self.decode(sample)

        return G