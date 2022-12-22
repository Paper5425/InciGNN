from __future__ import print_function

import copy
import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import subprocess
import collections
import igraph
import argparse
import pdb
import pygraphviz as pgv
import sys
from PIL import Image
import networkx as nx
import numpy as np
import torch.nn.functional as F
import math
import json
import scipy.sparse as sp
import pandas as pd
import pickle as pk
from stellargraph.data import EdgeSplitter
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
from torch_geometric.utils import to_undirected
from torchvision import transforms
import pygsp
import torch_geometric.transforms as T


# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

'''cora and citeseer'''


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _load_cora_to_graph(dataset='cora'):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    path = "./data/{}/".format(dataset)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    features = normalize_features(features)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(adj.shape[0])
    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    for e in edges:
        if e[0] != e[1]:
            g_stem.add_edge(e[0], e[1])

    return g_stem, features, labels


def _load_citeseer_to_graph():
    cs_content = pd.read_csv('./data/citeseer/citeseer.content', sep='\t', header=None)
    cs_cite = pd.read_csv('./data/citeseer/citeseer.cites', sep='\t', header=None)
    ct_idx = list(cs_content.index)
    paper_id = list(cs_content.iloc[:, 0])
    paper_id = [str(i) for i in paper_id]
    mp = dict(zip(paper_id, ct_idx))

    label = cs_content.iloc[:, -1]
    label = pd.get_dummies(label)

    feature = cs_content.iloc[:, 1:-1]

    mlen = cs_content.shape[0]
    adj = np.zeros((mlen, mlen))

    # adj = torch.FloatTensor(adj)
    features = torch.FloatTensor(np.array(feature))
    labels = torch.LongTensor(np.where(label)[1])

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(adj.shape[0])
    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    for i, j in zip(cs_cite[0], cs_cite[1]):
        if str(i) in mp.keys() and str(j) in mp.keys():
            x = mp[str(i)]
            y = mp[str(j)]
            if x != y:
                g_stem.add_edge(x, y)

    return g_stem, features, labels


def _load_wiki_to_graph(dataset_path='WikiCS'):
    print('Loading {} dataset...'.format(dataset_path))
    path = './data/{}/'.format(dataset_path)

    # load graph from file
    g_stem = load_object('./data/{}/graph.pk'.format(dataset_path))
    dataset = WikiCS(root=path)[0]

    features = dataset.x
    labels = dataset.y

    return g_stem, features, labels


def _load_WebKB_to_graph(dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    path = './data/{}/'.format(dataset)

    dataset = WebKB(root=path, name=dataset)[0]

    features = dataset.x
    labels = dataset.y
    edges = dataset.edge_index

    g_stem = igraph.Graph(directed=True)
    g_stem.add_vertices(labels.size(0))

    for i in range(g_stem.vcount()):
        g_stem.vs[i]['id'] = i
        g_stem.vs[i]['type'] = labels[i].tolist()
        g_stem.vs[i]['feature'] = features[i].tolist()

    edges_list = edges.t().tolist()

    for e in edges_list:
        if e[0] != e[1]:
            g_stem.add_edge(e[0], e[1])

    return g_stem, features, labels


def load_data(dataset="cora"):
    if dataset == 'cora':
        g_stem, features, labels = _load_cora_to_graph()
    elif dataset == 'citeseer':
        g_stem, features, labels = _load_citeseer_to_graph()
    elif dataset == 'WikiCS':
        g_stem, features, labels = _load_wiki_to_graph()

    adj = g_stem.get_adjacency().data
    adj = torch.FloatTensor(adj)
    assert adj.diag().sum() == 0

    edge_list = g_stem.get_edgelist()

    MAX_NODES = g_stem.vcount()
    MAX_EDGES = len(edge_list)
    MAX_EDGES_FALSE = MAX_EDGES

    ###############################################################
    # add some negative edges, which are not existed in the graph #
    ###############################################################
    # def ismember(a, b, tol=5):
    #     rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    #     return np.any(rows_close)
    #
    edge_false = []
    #
    # while len(edge_false) < MAX_EDGES_FALSE:
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], np.array(edge_list)):
    #         continue
    #     if edge_false:
    #         if ismember([idx_i, idx_j], np.array(edge_false)):
    #             continue
    #         if ismember([idx_j, idx_i], np.array(edge_false)):
    #             continue
    #     edge_false.append((idx_i, idx_j))
    #
    # # for i in range(MAX_NODES):
    # #     for j in range(MAX_NODES):
    # #         if i != j and (i, j) not in edge_list:
    # #             edge_false.append([i, j])
    #
    # assert ~ismember(edge_false, np.array(edge_list))
    # assert len(edge_list) == len(edge_false)

    ################################################################

    inci_mat_T = torch.zeros(MAX_EDGES * 1, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES * 1, MAX_NODES)

    inci_lb_T = torch.zeros(MAX_EDGES * 1, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES * 1, MAX_NODES)

    for i, e in enumerate(edge_list):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1

    adj = adj + adj.t()
    ft_ex = torch.zeros(features.size(0), 1)
    features = torch.cat([features, ft_ex], dim=-1)

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    graph_args.num_vertex_type = 6  # original types + start/end types
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES * 1

    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    # edge_list = edge_list + edge_false

    # edge_list = [list(e) for e in edge_list]

    edge_label = [1] * MAX_EDGES + [0] * MAX_EDGES_FALSE
    edge_label = torch.tensor(edge_label)

    return adj, features, labels, edge_list, edge_label, inci_mat_T, inci_mat_H, inci_lb_T, inci_lb_H, weight_tensor_T, weight_tensor_H, graph_args


def load_data_smooth(dataset="cora", add_false_Edge=False):
    if dataset == 'cora':
        g_stem, features, labels = _load_cora_to_graph()
        N_CLASS = 7
    elif dataset == 'citeseer':
        g_stem, features, labels = _load_citeseer_to_graph()
        N_CLASS = 6
    elif dataset == 'WikiCS':
        g_stem, features, labels = _load_wiki_to_graph()
        N_CLASS = 10
    elif dataset == 'Cornell':
        g_stem, features, labels = _load_WebKB_to_graph(dataset)
        N_CLASS = 5
    elif dataset == 'Texas':
        g_stem, features, labels = _load_WebKB_to_graph(dataset)
        N_CLASS = 5
    elif dataset == 'Wisconsin':
        g_stem, features, labels = _load_WebKB_to_graph(dataset)
        N_CLASS = 5

    adj = g_stem.get_adjacency().data
    edge_list = g_stem.get_edgelist()
    MAX_NODES = g_stem.vcount()
    adj = torch.FloatTensor(adj)
    assert adj.diag().sum() == 0

    adj = adj + torch.eye(len(adj)) + adj.t()
    deg = torch.sum(adj, dim=1)
    D = torch.diag(deg.pow(-0.5))
    adj = torch.mm(D, adj)
    adj = torch.mm(adj, D)

    MAX_EDGES = len(edge_list)

    ###############################################################
    # add some negative edges, which are not existed in the graph #
    ###############################################################
    edge_false = []

    if add_false_Edge:
        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        print('Generating false edges...')
        while len(edge_false) < MAX_EDGES:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])

            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], np.array(edge_list)):
                continue
            if edge_false:
                if ismember([idx_i, idx_j], np.array(edge_false)):
                    continue
                if ismember([idx_j, idx_i], np.array(edge_false)):
                    continue
            edge_false.append((idx_i, idx_j))

        assert ~ismember(edge_false, np.array(edge_list))
        assert len(edge_list) == len(edge_false)

    ################################################################

    MAX_EDGES_FALSE = len(edge_false)

    inci_mat_T = torch.zeros(MAX_EDGES + MAX_EDGES_FALSE, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES + MAX_EDGES_FALSE, MAX_NODES)

    for i, e in enumerate(edge_list):
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1

    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_mat_T.size())

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_mat_H.size())

    graph_args.num_vertex_type = N_CLASS
    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES + MAX_EDGES_FALSE
    graph_args.n_features = features.size(-1)

    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    edge_list = edge_list + edge_false
    edge_list = torch.tensor(edge_list)
    edge_label = torch.tensor([1] * len(edge_list) + [0] * len(edge_false))

    return adj, features, labels, edge_list, edge_label, inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, graph_args


def _list_len(l):
    length = []
    for item in l:
        length.append(len(item))
    return sum(length)


def _in_list(l, v):
    for item in l:
        if v in item:
            return True

    return False


##########################################
## inside funcs                         ##
##########################################
def _get_successors(v, G, max_n=7):
    pred = []
    v_ori = v
    max_n = max_n - 1
    ct = 0

    while _list_len(pred) != max_n:
        if ct == 100:
            return []

        v_temp = list(set(G.successors(v)))
        temp = v_temp.copy()

        for item in temp:
            if item == v_ori:
                v_temp.remove(item)
                continue

            if _in_list(pred, item):
                v_temp.remove(item)
                continue

        if len(v_temp) > (max_n - _list_len(pred)):
            v_temp = random.sample(v_temp, (max_n - _list_len(pred)))
            pred.append(v_temp)
        elif len(v_temp) == 0:
            if len(pred) > 0:
                v = random.sample(pred[-1], 1)[0]
            ct = ct + 1
            continue
        else:
            # print('PR:', v_list)
            pred.append(v_temp)
            v = random.sample(v_temp, 1)[0]

        if len(pred) == 0:
            return []

        ct = ct + 1

    return pred

def _get_successors_multi(v, G, max_n=7):
    pred_0 = []
    pred_1 = []
    v_ori = v
    max_n = max_n - 1
    ct = 0

    while _list_len(pred_0) != max_n:
        if ct == 100:
            break

        v_temp = list(set(G.successors(v)))
        temp = v_temp.copy()

        for item in temp:
            if item == v_ori:
                v_temp.remove(item)
                continue

            if _in_list(pred_0, item):
                v_temp.remove(item)
                continue

            if _in_list(pred_1, item):
                v_temp.remove(item)
                continue

        if len(v_temp) > (max_n - _list_len(pred_0)):
            v_temp = random.sample(v_temp, (max_n - _list_len(pred_0)))
            pred_0.append(v_temp)
            v_temp = random.sample(v_temp, (max_n - _list_len(pred_1)))
            pred_1.append(v_temp)

        elif len(v_temp) == 0:
            if len(pred_0) > 0:
                v = random.sample(pred_0[-1], 1)[0]
            ct = ct + 1
            continue
        else:
            # print('PR:', v_list)
            pred_0.append(v_temp)
            pred_1.append(v_temp)
            v = random.sample(v_temp, 1)[0]

        if len(pred_0) == 0:
            return []

        ct = ct + 1

    return [pred_0, pred_1]

def _get_predecessors(v, G, max_n=7):
    pred = []
    v_ori = v
    max_n = max_n - 1
    ct = 0

    while _list_len(pred) != max_n:
        if ct == 100:
            # pred = []
            break

        v_temp = list(set(G.predecessors(v)))
        temp = v_temp.copy()

        for item in temp:
            if item == v_ori:
                v_temp.remove(item)
                continue

            if _in_list(pred, item):
                v_temp.remove(item)
                continue

        if len(v_temp) > (max_n - _list_len(pred)):
            v_temp = random.sample(v_temp, (max_n - _list_len(pred)))
            pred.append(v_temp)
        elif len(v_temp) == 0:
            if len(pred) > 0:
                v = random.sample(pred[-1], 1)[0]
            ct = ct + 1
            continue
        else:
            # print('PR:', v_list)
            pred.append(v_temp)
            v = random.sample(v_temp, 1)[0]

            # v_list = list(set(v_list + list(G.neighbors(random.sample(v_list, 1)[0]))))
            # print('BA:', v_list)

        ct = ct + 1

    if len(pred) == 0:
        succ = _get_successors(v_ori, G, max_n=max_n + 1)

        if len(succ) == 0:
            g_sub_succ = []
        else:
            g_sub_succ = _get_subgraph_succ(v_ori, succ, G)

        return g_sub_succ

    g_sub = _get_subgraph(v_ori, pred, G)

    return g_sub

def _get_predecessors_multi(v, G, max_n=7):
    pred_0 = []
    pred_1 = []
    v_ori = v
    max_n = max_n - 1
    ct = 0

    while _list_len(pred_0) != max_n:
        if ct == 100:
            # pred = []
            break

        v_temp = list(set(G.predecessors(v)))
        temp = v_temp.copy()


        for item in temp:
            if item == v_ori:
                v_temp.remove(item)
                continue

            if _in_list(pred_0, item):
                v_temp.remove(item)
                continue

            if _in_list(pred_1, item):
                v_temp.remove(item)
                continue

        if len(v_temp) > (max_n - _list_len(pred_0)):
            v_temp_0 = random.sample(v_temp, (max_n - _list_len(pred_0)))
            pred_0.append(v_temp_0)
            v_temp_1 = random.sample(v_temp, (max_n - _list_len(pred_1)))
            pred_1.append(v_temp_1)

        elif len(v_temp) == 0:
            if len(pred_0) > 0:
                v = random.sample(pred_0[-1], 1)[0]
            ct = ct + 1
            continue
        else:
            pred_0.append(v_temp)
            pred_1.append(v_temp)
            v = random.sample(v_temp, 1)[0]

        ct = ct + 1

    if len(pred_0) == 0:
        succ = _get_successors_multi(v_ori, G, max_n=max_n + 1)

        if len(succ) == 0:
            g_sub_succ = []
        else:
            g_sub_succ_0 = _get_subgraph_succ(v_ori, succ[0], G)
            g_sub_succ_1 = _get_subgraph_succ(v_ori, succ[1], G)

        return [g_sub_succ_0, g_sub_succ_1]

    g_sub_0 = _get_subgraph(v_ori, pred_0, G)
    g_sub_1 = _get_subgraph(v_ori, pred_1, G)

    return [g_sub_0, g_sub_1]

def _get_subgraph(v, v_list, G):
    v_tmp = []
    for item in v_list:
        v_tmp = v_tmp + item
    v_tmp.append(v)
    sub_g = G.subgraph(v_tmp)
    e_list_ori = sub_g.get_edgelist()
    e_list = [(sub_g.vs[e[0]]['id'], sub_g.vs[e[1]]['id']) for e in e_list_ori]
    e_list_cp = e_list.copy()

    e_select = []
    e_copy = e_list.copy()
    for e in e_copy:
        if e[1] == v:
            e_select.append(e)
            e_list.remove(e)
            continue

        if len(v_list) > 1:
            for vs in v_list[:-1]:
                for vv in vs:
                    if e[1] == vv:
                        e_select.append(e)
                        e_list.remove(e)

    e_copy = e_list.copy()
    for e in e_copy:
        if e[0] in v_list[-1] and e[1] in v_list[-1]:
            if (e[1], e[0]) not in e_select:
                e_select.append(e)
                e_list.remove(e)

    for e in e_list:
        sub_g.delete_edges(e_list_ori[e_list_cp.index(e)])

    return sub_g

def _get_subgraph_succ(v, v_list, G):
    v_tmp = []
    for item in v_list:
        v_tmp = v_tmp + item
    v_tmp.append(v)
    sub_g = G.subgraph(v_tmp)
    e_list_ori = sub_g.get_edgelist()
    e_list = [(sub_g.vs[e[0]]['id'], sub_g.vs[e[1]]['id']) for e in e_list_ori]
    e_list_cp = e_list.copy()

    e_select = []
    e_copy = e_list.copy()
    for e in e_copy:
        if e[0] == v:
            e_select.append(e)
            e_list.remove(e)
            continue

        if len(v_list) > 1:
            for vs in v_list[:-1]:
                for vv in vs:
                    if e[0] == vv:
                        e_select.append(e)
                        e_list.remove(e)

    e_copy = e_list.copy()
    for e in e_copy:
        if e[0] in v_list[-1] and e[1] in v_list[-1]:
            if (e[1], e[0]) not in e_select:
                e_select.append(e)
                e_list.remove(e)

    for e in e_list:
        sub_g.delete_edges(e_list_ori[e_list_cp.index(e)])

    edges = sub_g.get_edgelist()
    sub_g.delete_edges()

    for e in edges:
        sub_g.add_edge(e[1], e[0])

    return sub_g

def _get_predecessors_inci(sub_g, v, max_n=7):
    v_count = max_n
    e_count = v_count * (v_count - 1)
    e_list = sub_g.get_edgelist()

    inci_mat_T = torch.zeros(e_count, v_count)
    inci_mat_H = torch.zeros(e_count, v_count)
    inci_lb_T = torch.zeros(e_count, v_count)
    inci_lb_H = torch.zeros(e_count, v_count)

    e_set = []
    for i in range(v_count):
        for j in range( v_count):
            if i != j:
                e_set.append([i, j])

    for i, e in enumerate(e_set):
        if (e[0], e[1]) in e_list:
            inci_mat_T[i][e[0]] = 1
            inci_mat_H[i][e[1]] = 1
            inci_lb_T[i][e[0]] = 1
            inci_lb_H[i][e[1]] = 1

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), \
           weight_tensor_T, weight_tensor_H, e_set


def _get_predecessors_attr(sub_g, v, max_n=7):
    vid = []
    label = []
    ft = []

    for i in range(sub_g.vcount()):
        vid.append(sub_g.vs[i]['id'])
        label.append(sub_g.vs[i]['type'])
        ft.append(sub_g.vs[i]['feature'])

        if sub_g.vs[i]['id'] == v:
            g_lb = sub_g.vs[i]['type']

    ft_pad = [[0] * len(sub_g.vs[i]['feature'])] * (max_n - sub_g.vcount())
    ft = ft + ft_pad

    lb_mask = [1] * len(label) + [0] * (max_n - len(label))

    lb_pad = [9] * (max_n - sub_g.vcount())
    label = label + lb_pad

    label = torch.tensor(label).unsqueeze(dim=0)
    ft = torch.tensor(ft).unsqueeze(dim=0)
    lb_mask = torch.tensor(lb_mask).unsqueeze(dim=0)
    g_lb = torch.tensor(g_lb).unsqueeze(dim=0)

    return g_lb, label, ft, lb_mask


def _get_predecessors_attr_mask(sub_g, v, max_n=7, test=None):
    vid = []
    label = []
    ft = []
    idx = []

    for i in range(sub_g.vcount()):
        vid.append(sub_g.vs[i]['id'])
        label.append(sub_g.vs[i]['type'])
        ft.append(sub_g.vs[i]['feature'])

        if sub_g.vs[i]['id'] == v:
            g_lb = sub_g.vs[i]['type']

        if sub_g.vs[i]['id'] in test:
           idx.append(i)

    ft_pad = [[0] * len(sub_g.vs[i]['feature'])] * (max_n - sub_g.vcount())
    ft = ft + ft_pad

    lb_mask = [1] * len(label) + [0] * (max_n - len(label))

    for i in idx:
        lb_mask[i] = 0

    lb_pad = [9] * (max_n - sub_g.vcount())
    label = label + lb_pad

    label = torch.tensor(label).unsqueeze(dim=0)
    ft = torch.tensor(ft).unsqueeze(dim=0)
    lb_mask = torch.tensor(lb_mask).unsqueeze(dim=0)
    g_lb = torch.tensor(g_lb).unsqueeze(dim=0)

    return g_lb, label, ft, lb_mask


def _get_predecessors_attr_ablation(sub_g, v, max_n=7, test=None):
    vid = []
    label = []
    ft = []
    idx = []

    for i in range(sub_g.vcount()):
        vid.append(sub_g.vs[i]['id'])
        label.append(sub_g.vs[i]['type'])
        ft.append(sub_g.vs[i]['feature'])

        if sub_g.vs[i]['id'] == v:
            g_lb = sub_g.vs[i]['type']

        if sub_g.vs[i]['id'] in test:
           idx.append(i)

    ft_pad = [[0] * len(sub_g.vs[i]['feature'])] * (max_n - sub_g.vcount())
    ft = ft + ft_pad

    lb_mask = [1] * len(label) + [0] * (max_n - len(label))

    for i in idx:
        lb_mask[i] = 0

    lb_pad = [9] * (max_n - sub_g.vcount())
    label = label + lb_pad

    label = torch.tensor(label).unsqueeze(dim=0)
    ft = torch.tensor(ft).unsqueeze(dim=0)
    lb_mask = torch.tensor(lb_mask).unsqueeze(dim=0)
    g_lb = torch.tensor(g_lb).unsqueeze(dim=0)

    p = max_n - sub_g.vcount()
    adj = sub_g.get_adjacency().data
    adj = torch.FloatTensor(adj)
    adj = F.pad(adj, [0, p, 0, p])

    return g_lb, label, ft, adj

def _get_subgraph_verts(v, G, v_test=None, max_n=7):
    v_list = []
    v_ori = v
    max_n = max_n - 1
    ct = 0

    while _list_len(v_list) != max_n:
        if ct == 100:
            return []

        v_temp = list(set(G.neighbors(v)))
        temp = v_temp.copy()

        for item in temp:
            if v_test != None:
                if item in v_test:
                    v_temp.remove(item)
                    continue

            if item == v_ori:
                v_temp.remove(item)
                continue

            if _in_list(v_list, item):
                v_temp.remove(item)
                continue

        if len(v_temp) > (max_n - _list_len(v_list)):
            v_temp = random.sample(v_temp, (max_n - _list_len(v_list)))
            v_list.append(v_temp)
        elif len(v_temp) == 0:
            ct = ct + 1
            continue
        else:
            v_list.append(v_temp)
            v = random.sample(v_temp, 1)[0]

        if len(v_list) == 0:
            return []

        ct = ct + 1

    v_n = []

    for item in v_list:
        v_n = v_n + item

    v_n.append(v_ori)

    return v_n

def _get_sub_attr(sub_g, v):
    vid = []
    label = []
    ft = []

    for i in range(sub_g.vcount()):
        vid.append(sub_g.vs[i]['id'])
        label.append(sub_g.vs[i]['type'])
        ft.append(sub_g.vs[i]['feature'])

        if sub_g.vs[i]['id'] == v:
            v_idx = i

    adj = sub_g.get_adjacency().data

    return vid, torch.tensor(label).unsqueeze(dim=0), torch.tensor(ft).unsqueeze(dim=0), torch.tensor(v_idx).unsqueeze(dim=0)

def _get_sub_inci(sub_g, max_n=7):
    v_count = sub_g.vcount()
    e_count = v_count * (v_count - 1)
    e_list = sub_g.get_edgelist()

    inci_mat_T = torch.zeros(e_count, v_count)
    inci_mat_H = torch.zeros(e_count, v_count)
    inci_lb_T = torch.zeros(e_count, v_count)
    inci_lb_H = torch.zeros(e_count, v_count)

    e_set = []
    for i in range(v_count):
        for j in range(v_count):
            if i != j:
                e_set.append([i, j])

    for i, e in enumerate(e_set):
        if (e[0], e[1]) in e_list:
            inci_mat_T[i][e[0]] = 1
            inci_mat_H[i][e[1]] = 1
            inci_lb_T[i][e[0]] = 1
            inci_lb_H[i][e[1]] = 1

    if inci_mat_T.size(1) != max_n:
        print(inci_mat_T.size())
        exit()

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    return inci_mat_T.unsqueeze(dim=0), inci_mat_H.unsqueeze(dim=0), inci_lb_T.unsqueeze(dim=0), inci_lb_H.unsqueeze(dim=0), \
           weight_tensor_T, weight_tensor_H, e_set


def _get_path_predecessors(v, G, max_n=7, max_p=5, test_set=[]):
    paths = []

    if v == 6788:
        a = 0

    v_ori = v
    # max_n = max_n - 1
    ct = 0

    while len(paths) < max_p:
        if ct == 100:
            break

        p_temp = []
        v = v_ori
        ctt = 0
        while len(p_temp) < max_n:
            if ctt == 100:
                break
            v_temp = list(set(G.predecessors(v)))
            v_temp_cp = v_temp.copy()

            if v_temp == []:
                if len(p_temp) > 0:
                    v = p_temp.pop()
                    ctt = ctt + 1
                    continue
                else:
                    break

            v_cur = random.sample(v_temp, 1)[0]
            v_temp.remove(v_cur)

            if v_cur == v_ori:
                ctt = ctt + 1
                continue

            p_temp.append(v)
            v = v_cur
            ctt = ctt + 1

        if len(p_temp) == max_n:
            paths.append(p_temp)

        ct = ct + 1

    if paths != []:
        ft = []

        while len(paths) > 0:
            v_list = paths.pop()
            ft_p = []
            while len(v_list) > 0:
                v_cur = v_list.pop()
                ft_p.append(G.vs[v_cur]['feature'])
            ft.append(ft_p)

        label = G.vs[v]['type']

        ft = torch.tensor(ft).unsqueeze(dim=0)
        label = torch.tensor(label).unsqueeze(dim=0)

        return (v_ori, ft, label)

    else:
        paths = _get_path_successors(v=v_ori, G=G, max_n=max_n, max_p=max_p, test_set=test_set)

    return paths

def _get_path_successors(v, G, max_n=7, max_p=5, test_set=[]):
    paths = []

    if v == 8117:
        a = 0

    v_ori = v
    # max_n = max_n - 1
    ct = 0

    while len(paths) < max_p:
        if ct == 100:
            break

        p_temp = []
        v = v_ori
        ctt = 0
        while len(p_temp) < max_n:
            if ctt == 100:
                break
            v_temp = list(set(G.successors(v)))
            v_temp_cp = v_temp.copy()

            if v_temp == []:
                if len(p_temp) > 0:
                    v = p_temp.pop()
                    ctt = ctt + 1
                    continue
                else:
                    break

            v_cur = random.sample(v_temp, 1)[0]
            v_temp.remove(v_cur)

            if v_cur == v_ori:
                ctt = ctt + 1
                continue

            p_temp.append(v)
            v = v_cur
            ctt = ctt + 1

        if len(p_temp) == max_n:
            paths.append(p_temp)

        ct = ct + 1

    if paths != []:
        ft = []

        while len(paths) > 0:
            v_list = paths.pop()
            ft_p = []
            while len(v_list) > 0:
                v_cur = v_list.pop()
                ft_p.append(G.vs[v_cur]['feature'])
            ft.append(ft_p)

        label = G.vs[v]['type']

        ft = torch.tensor(ft).unsqueeze(dim=0)
        label = torch.tensor(label).unsqueeze(dim=0)

        return (v_ori, ft, label)

    return paths


#################################################################################################
'''load and save objects'''

def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return


'''Data preprocessing'''
def load_ENAS_graphs(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    max_n_eg = 0  # maximum number of edges
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_ENAS_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)

            # elif fmt == 'mlp':
            #     g, n = decode_ENAS_to_
            max_n = max(max_n, n)
            g_list.append((g, y))
    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)

    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def load_ENAS_matrix(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []

    max_n = 0  # maximum number of nodes
    max_n_eg = 0  # maximum number of edges
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                # g, n = decode_ENAS_to_igraph(row)
                g = 0
                n = 0
                # adj_mat, adj_mat_N, ops, inci_lb_T, inci_lb_H, inci_mat_T, edge_t, weight_tensor, weight_tensor_T, weight_tensor_H, g
                adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_networkx(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            max_n_eg = max(max_n_eg, 0)

            g_list.append((y, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

    graph_args.num_vertex_type = n_types + 3  # original types + start/end types
    graph_args.max_n = 8  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def load_ENAS_matrix_MS(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []

    max_n = 0  # maximum number of nodes
    max_n_eg = 0  # maximum number of edges
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                # g, n = decode_ENAS_to_igraph(row)
                g = 0
                n = 0
                # adj_mat, adj_mat_N, ops, inci_lb_T, inci_lb_H, inci_mat_T, edge_t, weight_tensor, weight_tensor_T, weight_tensor_H, g
                adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_networkx_MS(row)

                g_lb, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            max_n_eg = max(max_n_eg, 0)

            g_list.append((y, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, g_lb, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

    graph_args.num_vertex_type = n_types + 3  # original types + start/end types
    graph_args.max_n = 8  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def load_ENAS_matrix_Full(name, n_types=6, fmt='igraph', link_pred=False, add_noise=False, rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []

    max_n = 8  # maximum number of nodes
    max_n_eg = 28  # maximum number of edges
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                # g, n = decode_ENAS_to_igraph(row)
                g = 0
                n = 0

                if link_pred:
                    adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_Full_masked(row, max_vert_n=max_n, max_edge_n=max_n_eg)
                    g_list.append((y, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))
                elif add_noise:
                    for _ in range(8):
                        adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_Full_noised(row, max_vert_n=max_n, max_edge_n=max_n_eg)
                        g_list.append((y, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))
                else:
                    adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_Full(row, max_vert_n=max_n, max_edge_n=max_n_eg)
                    g_list.append((torch.tensor([y]), adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)

    graph_args.num_vertex_type = n_types + 3  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


#######################################################
# load dataset for model pruning
#######################################################

def load_NN_models(dataset="cora"):

    g_list, max_n_node, max_n_edge = _load_ENAS_to_graph(dataset)
    N_CLASS = 8


    graph_args.num_vertex_type = N_CLASS
    graph_args.max_n = max_n_node
    graph_args.max_n_eg = max_n_edge
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1  # predefined end vertex type

    ng = int(len(g_list) * .9)

    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    return g_list[:ng], g_list[ng:], graph_args


def _load_ENAS_to_graph(name):#, n_types=6, fmt='igraph', link_pred=False, add_noise=False, rand_seed=0, with_y=True, burn_in=1000):
    g_list = []
    max_n_node = 0
    max_n_edge = 0

    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break

            row, y = eval(row)
            g, node_n = decode_ENAS_to_igraph(row)

            edge_index, ops, inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, edge_n = _get_graph_attributes(g, node_n)

            g_list.append((edge_index, ops, inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, y, g))

            max_n_node = max(max_n_node, node_n)
            max_n_edge = max(max_n_edge, edge_n)

    return g_list, max_n_node, max_n_edge


def _get_graph_attributes(g_stem, max_n=8):
    MAX_NODES = max_n

    ops = []

    for i in range(g_stem.vcount()):
        ops.append(g_stem.vs[i]['type'])

    edge_list = g_stem.get_edgelist()

    MAX_EDGES = len(edge_list)

    inci_mat_T = torch.zeros(MAX_EDGES, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES, MAX_NODES)

    for i, e in enumerate(edge_list):
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1

    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    ops = torch.tensor(ops) #.unsqueeze(dim=0)
    # inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    # inci_mat_H = inci_mat_H.unsqueeze(dim=0)
    edge_index = torch.tensor(edge_list).t()

    return edge_index, ops, inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, MAX_EDGES



#######################################################
# END of load data for mode pruing
#######################################################



###################################################
# load OGBG-Code dataset                          #
###################################################
def load_OGBG(num_vocab=5000, max_seq_len=5, emb_dim=300):
    MAX_NODES = 499
    MAX_EDGES = 1734

    dataset = PygGraphPropPredDataset(name='ogbg-code2')
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in range(len(dataset))], num_vocab)
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

    arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)

    filtered_dataset = []

    for g in dataset:
        if g.x.size(0) < 500:
            inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, edge_attr = process_OGBG(g)
            g.inci_T = inci_mat_T
            g.inci_H = inci_mat_H
            g.weight_T = weight_tensor_T
            g.weight_H = weight_tensor_H

            pad = torch.tensor([[98, 10030]] * (500-g.x.size(0))).long()
            g.x = torch.cat([g.x, pad], dim=0)

            g.node_depth = F.pad(g.node_depth, [0, 0, 0, 500-g.node_depth.size(0)])
            g.edge_attr = edge_attr

            filtered_dataset.append(g)

    graph_args.max_n = MAX_NODES  # maximum number of nodes
    graph_args.max_n_eg = MAX_EDGES

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))
    node_encoder = ASTNodeEncoder(emb_dim, num_nodetypes=len(nodetypes_mapping['type']), num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))
    ng = len(filtered_dataset)

    return filtered_dataset[:int(ng * 0.9)], filtered_dataset[int(ng * 0.9):], graph_args, node_encoder, arr_to_seq


def process_OGBG(g_ogb, max_vert_n=499, max_edge_n=1734):
    MAX_NODES = max_vert_n
    MAX_EDGES = max_edge_n

    edge_list = g_ogb.edge_index.t().tolist()
    edge_attr_list = g_ogb.edge_attr.tolist()

    # edge_idx = []
    # for i in range(MAX_NODES):
    #     for j in range(MAX_NODES):
    #         if i != j:
    #             edge_idx.append([i, j])

    inci_mat_T = torch.zeros(MAX_EDGES, MAX_NODES)
    inci_mat_H = torch.zeros(MAX_EDGES, MAX_NODES)

    inci_lb_T = torch.zeros(MAX_EDGES, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES, MAX_NODES)

    edge_attr = torch.zeros(MAX_EDGES, 2)

    # for i, e in enumerate(edge_idx):
    #     if e in edge_list:
    #         inci_mat_T[i][e[0]] = 1
    #         inci_mat_H[i][e[1]] = 1
    #         inci_lb_T[i][e[0]] = 1
    #         inci_lb_H[i][e[1]] = 1
    #         edge_attr[i,0] = edge_attr_list[edge_list.index(e)][0]
    #         edge_attr[i,1] = edge_attr_list[edge_list.index(e)][1]

    for i, e in enumerate(edge_list):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1
        inci_mat_T[i][e[0]] = 1
        inci_mat_H[i][e[1]] = 1

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    return inci_mat_T, inci_mat_H, weight_tensor_T, weight_tensor_H, edge_attr


def add_inci(g_ogb):
    if g_ogb.x.size(0) >= 500:
        return g_ogb

    edge_list = g_ogb.edge_index.t().tolist()

    MAX_NODES = 499
    MAX_EDGES = 1734

    inci_lb_T = torch.zeros(MAX_EDGES, MAX_NODES)
    inci_lb_H = torch.zeros(MAX_EDGES, MAX_NODES)

    # edge_attr = torch.zeros(MAX_EDGES, 2)

    for i, e in enumerate(edge_list):
        inci_lb_T[i][e[0]] = 1
        inci_lb_H[i][e[1]] = 1

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T
    weight_tensor_T = weight_tensor_T.view(inci_lb_T.size())

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H
    weight_tensor_H = weight_tensor_H.view(inci_lb_H.size())

    g_ogb.inci_T = inci_lb_T
    g_ogb.inci_H = inci_lb_H
    g_ogb.weight_T = weight_tensor_T
    g_ogb.weight_H = weight_tensor_H

    pad = torch.tensor([[98, 10030]] * (30 - g_ogb.x.size(0))).long()
    g_ogb.x = torch.cat([g_ogb.x, pad], dim=0)
    g_ogb.node_depth = F.pad(g_ogb.node_depth, [0, 0, 0, 30 - g_ogb.node_depth.size(0)])
    g_ogb.node_is_attributed = F.pad(g_ogb.node_is_attributed, [0, 0, 0, 499 - g_ogb.node_is_attributed.size(0)])

    return g_ogb


def filter(g_ogb):
    if g_ogb.x.size(0) < 500:
        return g_ogb

## EOS of OGBG ##


# load NAS-Bench-101 format NNs to ENAS, then transfor models trained on ENAS to NAS-Bench-101 directly
def load_ENAS_to_NASBench(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=10000):
    DUMMY = 'dummy'
    INPUT = 'input'
    OUTPUT = 'output'
    CONV1X1 = 'conv1x1-bn-relu'
    CONV3X3 = 'conv3x3-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    op2embeddings = {
        DUMMY: 0,
        INPUT: 1,
        OUTPUT: 2,
        CONV1X1: 5,
        CONV3X3: 3,
        MAXPOOL3X3: 8,
    }

    g_list = []

    max_n = 8
    max_n_eg = 28

    with open('data/%s.json' % name, 'r') as f:
        rows = json.load(f)
        for i, row in enumerate(tqdm(rows)):
            # if i > burn_in:
            #     continue

            if row is None:
                break
            adj_mat = row['matrix']
            ops = row['ops']
            ops = [op2embeddings[item] for item in ops]
            acc = row['test_accuracy']

            if fmt == 'igraph':
                g = 0
                n = 0
                adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_NASBench_to_networkx(adj_mat, ops, max_vert_n=max_n, max_edge_n=max_n_eg)

            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)

            g_list.append((torch.tensor([acc]), adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

    graph_args.num_vertex_type = n_types + 3  # original types + start/end types
    graph_args.max_n = 8  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.8)], g_list[int(ng * 0.8):], graph_args
    # return g_list, g_list, graph_args


DUMMY = 'dummy'
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

op2embeddings = {
    DUMMY: 0,
    INPUT: 1,
    OUTPUT: 2,
    CONV1X1: 3,
    CONV3X3: 4,
    MAXPOOL3X3: 5,
}


# load NAS-Bench-101 format NNs to networkx or tensors
def load_NASBench_matrix(name, n_types=6, fmt='igraph', link_pred=False, add_noise=False, rand_seed=0, with_y=True, burn_in=1000):
    g_list = []

    max_n = 7
    max_n_eg = 21

    with open('data/%s.json' % name, 'r') as f:
        rows = json.load(f)
        for i, row in enumerate(tqdm(rows)):
            # if i > burn_in:
            #     continue

            if row is None:
                break
            adj_mat = row['matrix']
            ops = row['ops']
            ops = [op2embeddings[item] for item in ops]
            acc = row['test_accuracy']

            if fmt == 'igraph':
                g = 0
                n = 0

                if link_pred:
                    adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_NASBench_to_networkx_masked(adj_mat, ops, max_vert_n=max_n, max_edge_n=max_n_eg)
                elif add_noise:
                    adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_NASBench_to_networkx_noised(adj_mat, ops, max_vert_n=max_n, max_edge_n=max_n_eg)
                else:
                    adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_NASBench_to_networkx(adj_mat, ops, max_vert_n=max_n, max_edge_n=max_n_eg)

            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)

            g_list.append((torch.tensor([acc]), adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

    graph_args.num_vertex_type = 6  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.max_n_eg = max_n_eg
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


# load NAS-Bench-101 format NNs to networkx or tensors
def load_NASBench_matrix_MS(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    g_list = []
    with open('data/%s.json' % name, 'r') as f:
        rows = json.load(f)
        for i, row in enumerate(tqdm(rows)):
            # if i > burn_in:
            #     continue

            if row is None:
                break
            adj_mat = row['matrix']
            ops = row['ops']
            ops = [op2embeddings[item] for item in ops]
            acc = row['test_accuracy']

            if fmt == 'igraph':
                adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_NASBench_to_networkx(adj_mat, ops)

                g_lb, n = decode_ENAS_to_tensor(row, n_types)

            g_list.append((acc, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, g_lb, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

    graph_args.num_vertex_type = 6  # original types + start/end types
    graph_args.max_n = 7  # maximum number of nodes
    graph_args.max_n_eg = 21
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d, maximum # edges: %d' % (graph_args.max_n, graph_args.max_n_eg))

    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def load_NASBench_graphs(name, n_types=3, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('data/%s.txt' % name, 'r') as f:
        rows = json.load(f)

        for i, row in enumerate(tqdm(rows)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                adj_mat = row['matrix']
                ops = row['ops']
                ops = [op2embeddings[item] for item in ops]
                y = row['test_accuracy']
                g, n = decode_NASBench_to_igraph(adj_mat, ops)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            g_list.append((g, y))

    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 1  # predefined start vertex type
    graph_args.END_TYPE = 2  # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def decode_ENAS_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # assign 2, 3, ... to other types
        type_feature = one_hot(node_type, n_types)
        if i == 0:
            edge_feature = torch.zeros(1, n + 1)  # a node will have at most n+1 connections
        else:
            edge_feature = torch.cat([torch.FloatTensor(node[1:]).unsqueeze(0), torch.zeros(1, n + 1 - i)], 1)  # pad zeros
        edge_feature[0, i] = 1  # ENAS node always connects from the previous node
        g.append(torch.cat([type_feature, edge_feature], 1))
    # process the output node43219463225958843, Edge error: 0.0000
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n + 1)
    edge_feature[0, n] = 1  # output node only connects from the final node in ENAS
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n + 2


def decode_ENAS_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n + 2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 2  # assign 2, 3, ... to other types
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)
    g.vs[n + 1]['type'] = 1  # output node
    g.add_edge(n, n + 1)
    # note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n + 2


# ENAS数据集，最大节点数为8，最大边数为20
# 为了计算一致，这里统一用0进行补齐
# 定义，1为输入节点编码，2为输出结点编码，于是，占了三位，其余节点往后推3位即可
MAX_NODES = 8
MAX_EDGES = 20


def decode_ENAS_to_networkx(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edges_t = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 1  # input node
    ops.append(1)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 3  # assign 3, 4, ... to other types
        ops.append(node[0] + 3)
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)

    g.vs[n + 1]['type'] = 2  # output node
    g.add_edge(n, n + 1)
    ops.append(2)

    adj_mat = np.array(g.get_adjacency().data)
    g_nx = nx.DiGraph(adj_mat)

    edges = list(g_nx.edges)
    nodes = list(g_nx.nodes)

    for (src, trg) in edges:
        edges_t.append([ops[src], ops[trg]])

    inci_mat_nx = nx.incidence_matrix(g_nx, oriented=True).toarray()
    adj_mat = torch.from_numpy(adj_mat).float()

    inci_matt = torch.from_numpy(inci_mat_nx).float()
    inci_mat_T = torch.mm(inci_matt.transpose(0, 1), inci_matt) - torch.eye(inci_matt.size(1)) * 2

    inci_lb_T = torch.zeros_like(inci_matt)
    inci_lb_H = torch.zeros_like(inci_matt)
    inci_lb_T[inci_matt == -1] = 1.
    inci_lb_H[inci_matt == 1] = 1.

    n_edges = len(edges)
    if n_edges < MAX_EDGES:
        diff = MAX_EDGES - n_edges
        # 为了补齐长度，需要增加一些原本就不存在的边，那么这些边我们都统一加在最后一个节点上，方向就是他自己指向他自己
        edges = edges + [[7, 7]] * diff
        edges_t = edges_t + [[0, 0]] * diff
        inci_mat_T = F.pad(inci_mat_T, [0, diff, 0, diff])
        # inci_matt = F.pad(inci_matt, [0, diff])
        inci_lb_T = F.pad(inci_lb_T, [0, diff])
        inci_lb_H = F.pad(inci_lb_H, [0, diff])

    n_ops = len(ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES - n_ops
        nodes = nodes + np.arange(n_ops, MAX_NODES).tolist()
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])
        inci_lb_T = F.pad(inci_lb_T, [0, 0, 0, diff])
        inci_lb_H = F.pad(inci_lb_H, [0, 0, 0, diff])

        for i in range(1, diff + 1):
            inci_lb_T[-i][-1] = 1
            inci_lb_H[-i][-1] = 1

    adj_mat_L = adj_mat
    adj_mat = adj_mat + adj_mat.transpose(0, 1) + torch.eye(adj_mat.size(0))
    adj_mat_N = 0.5 * (adj_mat + adj_mat.transpose(0, 1)) + torch.eye(adj_mat.size(0))
    inci_mat_T = inci_mat_T + torch.eye(inci_mat_T.size(0))

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    # nm = adj_mat.shape[0] * adj_mat.shape[0] / float((adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) * 2)

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    nodes = torch.tensor(nodes).unsqueeze(dim=0)
    adj_mat_L = adj_mat_L.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_lb_T.unsqueeze(dim=0)
    inci_lb_H = inci_lb_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edges = torch.tensor(edges).unsqueeze(dim=0)
    edges_t = torch.tensor(edges_t).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat_L, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g


# ENAS数据集，最大节点数为8，最大边数为20
# 因为会将边进行初始化，且不用点来表示边
# 于是，生成一个全连接的图，主要是边
# 为了计算，需要将原Incidence矩阵扩展成一个全连接的矩阵
# 如果是8个节点的全连接图，边数为28
def decode_ENAS_to_Full(row, max_vert_n=8, max_edge_n=28):
    MAX_NODES = max_vert_n
    MAX_EDGES = max_edge_n

    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edge_idx = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 1  # input node
    ops.append(1)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 3  # assign 3, 4, ... to other types
        ops.append(node[0] + 3)
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)

    g.vs[n + 1]['type'] = 2  # output node
    g.add_edge(n, n + 1)
    ops.append(2)

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    n_ops = len(ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_mat_H = torch.zeros(MAX_NODES, MAX_EDGES)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            edge_idx.append((i, j))

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(8).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_mat_T.unsqueeze(dim=0)
    inci_lb_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edges = torch.tensor(edges).unsqueeze(dim=0)
    edges_t = torch.arange(28).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, ops, ops, ops, inci_lb_T, inci_lb_H, ops, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g

def decode_ENAS_to_Full_masked(row, max_vert_n, max_edge_n):
    MAX_NODES = max_vert_n
    MAX_EDGES = max_edge_n

    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edge_idx = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 1  # input node
    ops.append(1)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 3  # assign 3, 4, ... to other types
        ops.append(node[0] + 3)
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)

    g.vs[n + 1]['type'] = 2  # output node
    g.add_edge(n, n + 1)
    ops.append(2)

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    n_ops = len(ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_mat_H = torch.zeros(MAX_NODES, MAX_EDGES)

    inci_lb_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_lb_H = torch.zeros(MAX_NODES, MAX_EDGES)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            edge_idx.append((i, j))

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_lb_T[i, idx] = 1
                inci_lb_H[j, idx] = 1

    edge_positive = random.sample(edges, 1)[0]

    while True:
        edge_negative = random.sample(edge_idx, 1)[0]
        if edge_negative not in edges:
            break

    edges.remove(edge_positive)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(MAX_NODES).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    inci_mat_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat = inci_mat_T.unsqueeze(dim=0)
    edge_positive = torch.tensor(edge_positive).unsqueeze(dim=0)
    edge_negative = torch.tensor(edge_negative).unsqueeze(dim=0)
    inci_lb_T = inci_lb_T.unsqueeze(dim=0)
    inci_lb_H = inci_lb_H.unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    # return adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edge_positive, edge_negative, weight_tensor, weight_tensor_T, weight_tensor_H, g
    return adj_mat, edge_positive, edge_negative, ops, inci_mat_T, inci_mat_H, inci_mat_T, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g

def decode_ENAS_to_Full_noised(row, max_vert_n, max_edge_n):
    MAX_NODES = max_vert_n
    MAX_EDGES = max_edge_n

    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edge_idx = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 1  # input node
    ops.append(1)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 3  # assign 3, 4, ... to other types
        ops.append(node[0] + 3)
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)

    g.vs[n + 1]['type'] = 2  # output node
    g.add_edge(n, n + 1)
    ops.append(2)

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    n_ops = len(ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_mat_H = torch.zeros(MAX_NODES, MAX_EDGES)

    inci_lb_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_lb_H = torch.zeros(MAX_NODES, MAX_EDGES)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            edge_idx.append((i, j))

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_lb_T[i, idx] = 1
                inci_lb_H[j, idx] = 1

    edge_positive = random.sample(edges, 1)[0]

    while True:
        edge_negative = random.sample(edge_idx, 1)[0]
        if edge_negative not in edges:
            break

    edges.remove(edge_positive)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[1] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[1] - inci_lb_H.sum()) / inci_lb_H.sum()


    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(MAX_NODES).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    inci_mat_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat = inci_mat_T.unsqueeze(dim=0)
    edge_positive = torch.tensor(edge_positive).unsqueeze(dim=0)
    edge_negative = torch.tensor(edge_negative).unsqueeze(dim=0)
    inci_lb_T = inci_lb_T.unsqueeze(dim=0)
    inci_lb_H = inci_lb_H.unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, edge_positive, edge_negative, ops, inci_mat_T, inci_mat_H, adj_mat_N, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g


MAX_NODES = 8
MAX_EDGES = 20
def decode_ENAS_to_networkx_MS(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edges_t = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 1  # input node
    ops.append(1)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 3  # assign 3, 4, ... to other types
        ops.append(node[0] + 3)
        g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i + 1)

    g.vs[n + 1]['type'] = 2  # output node
    g.add_edge(n, n + 1)
    ops.append(2)

    adj_mat = np.array(g.get_adjacency().data)
    g_nx = nx.DiGraph(adj_mat)

    edges = list(g_nx.edges)
    nodes = list(g_nx.nodes)

    for (src, trg) in edges:
        edges_t.append([ops[src], ops[trg]])

    inci_mat_nx = nx.incidence_matrix(g_nx, oriented=True).toarray()
    adj_mat = torch.from_numpy(adj_mat).float()

    inci_matt = torch.from_numpy(inci_mat_nx).float()
    inci_mat_T = torch.mm(inci_matt.transpose(0, 1), inci_matt) - torch.eye(inci_matt.size(1)) * 2

    inci_lb_T = torch.zeros_like(inci_matt)
    inci_lb_H = torch.zeros_like(inci_matt)
    inci_lb_T[inci_matt == -1] = 1.
    inci_lb_H[inci_matt == 1] = 1.

    n_edges = len(edges)
    if n_edges < MAX_EDGES:
        diff = MAX_EDGES - n_edges
        # 为了补齐长度，需要增加一些原本就不存在的边，那么这些边我们都统一加在最后一个节点上，方向就是他自己指向他自己
        edges = edges + [[7, 7]] * diff
        edges_t = edges_t + [[0, 0]] * diff
        inci_mat_T = F.pad(inci_mat_T, [0, diff, 0, diff])
        inci_lb_T = F.pad(inci_lb_T, [0, diff])
        inci_lb_H = F.pad(inci_lb_H, [0, diff])

    n_ops = len(ops)
    if n_ops < MAX_NODES:
        diff = MAX_NODES - n_ops
        nodes = nodes + np.arange(n_ops, MAX_NODES).tolist()
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])
        inci_lb_T = F.pad(inci_lb_T, [0, 0, 0, diff])
        inci_lb_H = F.pad(inci_lb_H, [0, 0, 0, diff])

        for i in range(1, diff + 1):
            inci_lb_T[-i][-1] = 1
            inci_lb_H[-i][-1] = 1

    adj_mat = 0.5 * (adj_mat + adj_mat.t()) + torch.eye(adj_mat.size(0))

    D_T = torch.mm(inci_lb_T, inci_lb_T.t()) + torch.eye(inci_lb_T.size(0))
    D_T = torch.diag(torch.pow(torch.sum(D_T, dim=0), -0.5))

    D_H = torch.mm(inci_lb_H, inci_lb_H.t()) + torch.eye(inci_lb_H.size(0))
    D_H = torch.diag(torch.pow(torch.sum(D_H, dim=0), -0.5))

    adj_T = torch.mm(torch.mm(D_T, adj_mat), D_T)
    adj_H = torch.mm(torch.mm(D_H, adj_mat), D_H)

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()

    pos_weight_T = float(inci_lb_T.shape[0] * inci_lb_T.shape[0] - inci_lb_T.sum()) / inci_lb_T.sum()
    pos_weight_H = float(inci_lb_H.shape[0] * inci_lb_H.shape[0] - inci_lb_H.sum()) / inci_lb_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_lb_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_lb_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    nodes = torch.tensor(nodes).unsqueeze(dim=0)
    adj_mat_L = adj_T.unsqueeze(dim=0)
    adj_mat_N = adj_H.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_lb_T.unsqueeze(dim=0)
    inci_lb_H = inci_lb_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edges = torch.tensor(edges).unsqueeze(dim=0)
    edges_t = torch.tensor(edges_t).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat_L, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g


def decode_NASBench_to_igraph(adj_mat, ops):
    n_ops = len(ops)
    g = igraph.Graph.Adjacency(adj_mat, mode='directed')

    for i, node in enumerate(ops):
        g.vs[i]['type'] = node  # assign 3, 4, ... to other types

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n_ops


# NASBench数据集，最大节点数为7，最大边数为9，但是这里改为全连接网络，所以共有21条边
# 为了计算一致，这里统一用0进行补齐
# 定义，1为输入节点编码，2为输出结点编码，于是，占了三位，其余节点往后推3位即可
NAS_MAX_NODES = 7
NAS_MAX_EDGES = 21


# def decode_NASBench_to_networkx(adj_mat, ops):
#     edges_t = []
#     n_ops = len(ops)
#     g = igraph.Graph.Adjacency(adj_mat, mode='directed')
#
#     for i, node in enumerate(ops):
#         g.vs[i]['type'] = node  # assign 3, 4, ... to other types
#
#     adj_mat = np.array(g.get_adjacency().data)
#     edges = g.get_edgelist()
#
#     adj_mat = torch.from_numpy(adj_mat).float()
#
#     if n_ops < NAS_MAX_NODES:
#         diff = NAS_MAX_NODES - n_ops
#         ops = ops + [0] * diff
#         adj_mat = F.pad(adj_mat, [0, diff, 0, diff])
#
#     inci_mat_T = torch.zeros(7, 21)
#     inci_mat_H = torch.zeros(7, 21)
#
#     edge_idx = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
#
#     for i in range(6):
#         for j in range(i + 1, 7):
#             if (i, j) in edges:
#                 idx = edge_idx.index((i, j))
#                 inci_mat_T[i, idx] = 1
#                 inci_mat_H[j, idx] = 1
#
#     pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
#     pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
#     pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()
#
#     weight_mask = adj_mat.view(-1) == 1
#     weight_tensor = torch.ones(weight_mask.size(0))
#     weight_tensor[weight_mask] = pos_weight
#
#     weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
#     weight_tensor_T = torch.ones(weight_mask_T.size(0))
#     weight_tensor_T[weight_mask_T] = pos_weight_T
#
#     weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
#     weight_tensor_H = torch.ones(weight_mask_H.size(0))
#     weight_tensor_H[weight_mask_H] = pos_weight_H
#
#     adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))
#
#     # calculate DAD
#     D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
#     adj_mat_N = torch.mm(D, adj_mat_N)
#     adj_mat_N = torch.mm(adj_mat_N, D)
#
#     adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))
#
#     nodes = torch.arange(7).unsqueeze(dim=0)
#     adj_mat = adj_mat.unsqueeze(dim=0)
#     adj_mat_N = adj_mat_N.unsqueeze(dim=0)
#     ops = torch.tensor(ops).unsqueeze(dim=0)
#     inci_lb_T = inci_mat_T.unsqueeze(dim=0)
#     inci_lb_H = inci_mat_H.unsqueeze(dim=0)
#     inci_mat_T = inci_mat_T.unsqueeze(dim=0)
#     edges = torch.tensor(edges).unsqueeze(dim=0)
#     edges_t = torch.arange(21).unsqueeze(dim=0)
#
#     # note that the nodes 0, 1, ... n+1 are in a topological order
#     return adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g


def decode_NASBench_to_networkx_masked(adj_mat, ops, max_vert_n, max_edge_n):
    NAS_MAX_NODES = max_vert_n
    NAS_MAX_EDGES = max_edge_n
    edge_idx = []
    edges_t = []
    n_ops = len(ops)
    g = igraph.Graph.Adjacency(adj_mat, mode='directed')

    for i, node in enumerate(ops):
        g.vs[i]['type'] = node  # assign 3, 4, ... to other types

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    if n_ops < NAS_MAX_NODES:
        diff = NAS_MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)
    inci_mat_H = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            edge_idx.append((i, j))

    edge_positive = random.sample(edges, 1)[0]

    while True:
        edge_negative = random.sample(edge_idx, 1)[0]
        if edge_negative not in edges:
            break

    edges.remove(edge_positive)

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(7).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_mat_T.unsqueeze(dim=0)
    inci_lb_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edge_positive = torch.tensor(edge_positive).unsqueeze(dim=0)
    edge_negative = torch.tensor(edge_negative).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edge_positive, edge_negative, weight_tensor, weight_tensor_T, weight_tensor_H, g


def decode_NASBench_to_networkx_noised(adj_mat, ops, max_vert_n, max_edge_n):
    NAS_MAX_NODES = max_vert_n
    NAS_MAX_EDGES = max_edge_n
    edge_idx = []
    edges_t = []
    n_ops = len(ops)
    g = igraph.Graph.Adjacency(adj_mat, mode='directed')

    for i, node in enumerate(ops):
        g.vs[i]['type'] = node  # assign 3, 4, ... to other types

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    if n_ops < NAS_MAX_NODES:
        diff = NAS_MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)
    inci_mat_H = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)

    inci_lb_T = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)
    inci_lb_H = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            edge_idx.append((i, j))

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_lb_T[i, idx] = 1
                inci_lb_H[j, idx] = 1

    edge_positive = random.sample(edges, 1)[0]

    while True:
        edge_negative = random.sample(edge_idx, 1)[0]
        if edge_negative not in edges:
            break

    edges.remove(edge_positive)

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(7).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    inci_mat_H = inci_mat_H.unsqueeze(dim=0)
    # inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edge_positive = torch.tensor(edge_positive).unsqueeze(dim=0)
    edge_negative = torch.tensor(edge_negative).unsqueeze(dim=0)
    inci_lb_T = inci_lb_T.unsqueeze(dim=0)
    inci_lb_H = inci_lb_H.unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, adj_mat_N, nodes, ops, inci_mat_T, inci_mat_H, inci_mat_T, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g


# NASBench数据集，最大节点数为7，最大边数为9，
# 但是为了直接迁移ENAS训练好的模型，这里填充成最大节点8，最大边数28
# 为了计算一致，这里统一用0进行补齐
# 定义，1为输入节点编码，2为输出结点编码，于是，占了三位，其余节点往后推3位即可
def decode_NASBench_to_networkx(adj_mat, ops, max_vert_n, max_edge_n):
    NAS_MAX_NODES = max_vert_n
    NAS_MAX_EDGES = max_edge_n
    edges_t = []
    edge_idx = []
    n_ops = len(ops)
    g = igraph.Graph.Adjacency(adj_mat, mode='directed')

    for i, node in enumerate(ops):
        g.vs[i]['type'] = node  # assign 3, 4, ... to other types

    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    if n_ops < NAS_MAX_NODES:
        diff = NAS_MAX_NODES - n_ops
        ops = ops + [0] * diff
        adj_mat = F.pad(adj_mat, [0, diff, 0, diff])

    inci_mat_T = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)
    inci_mat_H = torch.zeros(NAS_MAX_NODES, NAS_MAX_EDGES)

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            edge_idx.append((i, j))

    for i in range(NAS_MAX_NODES - 1):
        for j in range(i + 1, NAS_MAX_NODES):
            if (i, j) in edges:
                idx = edge_idx.index((i, j))
                inci_mat_T[i, idx] = 1
                inci_mat_H[j, idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(NAS_MAX_NODES).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_mat_T.unsqueeze(dim=0)
    inci_lb_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edges = torch.tensor(edges).unsqueeze(dim=0)
    edges_t = torch.arange(NAS_MAX_EDGES).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g


def flat_ENAS_to_nested(row, n_nodes):
    # transform a flattened ENAS string to a nested list of ints
    if type(row) == str:
        row = [int(x) for x in row.split()]
    cnt = 0
    res = []
    for i in range(1, n_nodes + 1):
        res.append(row[cnt:cnt + i])
        cnt += i
        if cnt == len(row):
            break
    return res


# def decode_igraph_to_ENAS(g):
#     # decode an igraph to a flattend ENAS string
#     n = g.vcount()
#     res = []
#     adjlist = g.get_adjlist(igraph.IN)
#     for i in range(1, n - 1):
#         res.append(int(g.vs[i]['type']) - 2)
#         row = [0] * (i - 1)
#         for j in adjlist[i]:
#             if j < i - 1:
#                 row[j] = 1
#         res += row
#
#     return ' '.join(str(x) for x in res)

def decode_igraph_to_ENAS(g):
    res = []
    adj_mat = np.array(g.get_adjacency().data)

    for i in range(adj_mat.shape[0] - 1):
        for j in range(i + 1, adj_mat.shape[1]):
            res.append(adj_mat[i][j])

    for i in range(g.vcount()):
        res.append(g.vs[i]['type'].item())

    return ' '.join(str(x) for x in res)


'''
# some code to test format transformations
row = '[[4], [0, 1], [3, 1, 0], [3, 0, 1, 1], [1, 1, 1, 1, 1], [2, 1, 1, 0, 1, 1], [5, 1, 1, 1, 1, 1, 0], [2, 0, 0, 1, 0, 0, 1, 0]]'
row = '[[2], [2, 0], [4, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0, 1], [3, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 0], [4, 0, 0, 0, 0, 0, 0, 0], [4, 1, 0, 0, 1, 0, 0, 0, 0], [3, 0, 1, 1, 0, 0, 1, 0, 0, 0], [5, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]'
g, _ = decode_ENAS_to_igraph(row)
string = decode_igraph_to_ENAS(g)
print(row, string)
pdb.set_trace()
pwd = os.getcwd()
os.chdir('software/enas/')
os.system('./scripts/custom_cifar10_macro_final.sh ' + '"' + string + '"')
os.chdir(pwd)
'''


def load_BN_graphs(name, n_types=8, fmt='igraph', rand_seed=0, with_y=True):
    # load raw Bayesian network strings to igraphs or tensors
    g_list = []
    max_n = 0
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_BN_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_BN_to_tensor(row, n_types)
            max_n = max(max_n, n)
            assert (max_n == n)  # all BNs should have the same node number
            g_list.append((g, y))

    graph_args.num_class = 1  # how many classes of graphs
    graph_args.num_vertex_type = n_types + 2  # how many vertex types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1  # predefined end vertex type
    ng = len(g_list)
    print('# classes: %d' % graph_args.num_class)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def load_BN_matrix_Full(name, n_types=8, fmt='igraph', rand_seed=0, with_y=True):
    # load raw Bayesian network strings to igraphs or tensors
    g_list = []
    max_n = 10
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_BN_to_Full(row)
                g_list.append((torch.tensor([y]), adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))

            elif fmt == 'string':
                g, n = decode_BN_to_tensor(row, n_types)
            # max_n = max(max_n, n)
            # assert (max_n == n)  # all BNs should have the same node number
            # g_list.append((g, y))

    graph_args.num_class = 1  # how many classes of graphs
    graph_args.num_vertex_type = n_types + 2  # how many vertex types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.max_n_eg = 45
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1  # predefined end vertex type
    ng = len(g_list)
    print('# classes: %d' % graph_args.num_class)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def decode_BN_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    end_vertices = [True] * n
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # convert 0, 1, 2... to 2, 3, 4...
        type_feature = one_hot(node_type, n_types)
        edge_feature = torch.zeros(1, n + 1)  # a node will have at most n+1 connections
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            edge_feature[0, 0] = 1
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    edge_feature[0, j + 1] = 1
                    end_vertices[j] = False
        g.append(torch.cat([type_feature, edge_feature], 1))
    # output node
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n + 1)
    for j, flag in enumerate(end_vertices):  # connect all loose-end vertices to the output node
        if flag == True:
            edge_feature[0, j + 1] = 1
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n + 2


def decode_BN_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n + 2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        cur_type = node[0] + 2
        g.vs[i + 1]['type'] = cur_type

        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            g.add_edge(0, i + 1)
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    g.add_edge(j + 1, i + 1)
    g.vs[n + 1]['type'] = 1  # output node
    end_vertices = [v.index for v in g.vs.select(_outdegree_eq=0) if v.index != n + 1]
    for j in end_vertices:  # connect all loose-end vertices to the output node
        g.add_edge(j, n + 1)
    return g, n + 2


def decode_BN_to_Full(row, max_vert_n=10, max_edge_n=45):
    MAX_NODES = max_vert_n
    MAX_EDGES = max_edge_n

    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    ops = []
    edge_idx = []

    g.add_vertices(n + 2)
    g.vs[0]['type'] = 0  # input node
    ops.append(0)

    for i, node in enumerate(row):
        g.vs[i + 1]['type'] = node[0] + 2  # assign 3, 4, ... to other types
        ops.append(node[0] + 2)

        if sum(node[1:]) == 0:
            g.add_edge(0, i + 1)
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    g.add_edge(j + 1, i + 1)

    g.vs[n + 1]['type'] = 1  # output node
    ops.append(1)

    end_vertices = [v.index for v in g.vs.select(_outdegree_eq=0) if v.index != n+1]
    for j in end_vertices:
        g.add_edge(j, n + 1)


    adj_mat = np.array(g.get_adjacency().data)
    edges = g.get_edgelist()

    adj_mat = torch.from_numpy(adj_mat).float()

    n_ops = len(ops)
    assert n_ops == MAX_NODES

    inci_mat_T = torch.zeros(MAX_NODES, MAX_EDGES)
    inci_mat_H = torch.zeros(MAX_NODES, MAX_EDGES)

    for i in range(MAX_NODES - 1):
        for j in range(i + 1, MAX_NODES):
            edge_idx.append((i, j))

    # for i in range(MAX_NODES - 1):
    #     for j in range(i + 1, MAX_NODES):
    for item in edge_idx:
        if item in edges:
            idx = edge_idx.index(item)
            inci_mat_T[item[0], idx] = 1
            inci_mat_H[item[1], idx] = 1

    pos_weight = float(adj_mat.shape[0] * adj_mat.shape[0] - adj_mat.sum()) / adj_mat.sum()
    pos_weight_T = float(inci_mat_T.shape[0] * inci_mat_T.shape[1] - inci_mat_T.sum()) / inci_mat_T.sum()
    pos_weight_H = float(inci_mat_H.shape[0] * inci_mat_H.shape[1] - inci_mat_H.sum()) / inci_mat_H.sum()

    weight_mask = adj_mat.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    weight_mask_T = inci_mat_T.contiguous().view(-1) == 1
    weight_tensor_T = torch.ones(weight_mask_T.size(0))
    weight_tensor_T[weight_mask_T] = pos_weight_T

    weight_mask_H = inci_mat_H.contiguous().view(-1) == 1
    weight_tensor_H = torch.ones(weight_mask_H.size(0))
    weight_tensor_H[weight_mask_H] = pos_weight_H

    adj_mat_N = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    # calculate DAD
    D = torch.diag(torch.sum(adj_mat_N, dim=1).pow(-0.5))
    adj_mat_N = torch.mm(D, adj_mat_N)
    adj_mat_N = torch.mm(adj_mat_N, D)

    adj_mat = adj_mat + adj_mat.t() + torch.eye(adj_mat.size(0))

    nodes = torch.arange(8).unsqueeze(dim=0)
    adj_mat = adj_mat.unsqueeze(dim=0)
    adj_mat_N = adj_mat_N.unsqueeze(dim=0)
    ops = torch.tensor(ops).unsqueeze(dim=0)
    inci_lb_T = inci_mat_T.unsqueeze(dim=0)
    inci_lb_H = inci_mat_H.unsqueeze(dim=0)
    inci_mat_T = inci_mat_T.unsqueeze(dim=0)
    edges = torch.tensor(edges).unsqueeze(dim=0)
    edges_t = torch.arange(28).unsqueeze(dim=0)

    # note that the nodes 0, 1, ... n+1 are in a topological order
    return adj_mat, ops, ops, ops, inci_lb_T, inci_lb_H, ops, inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g


def decode_igraph_to_BN_adj(g):
    # decode an BN igraph to its flattened adjacency matrix string
    types = g.vs['type'][1:-1]
    real_order = np.argsort(types).tolist()
    adj = np.array(g.get_adjacency().data)[1:-1, 1:-1]
    adj = adj[real_order][:, real_order]
    return ' '.join(str(x) for x in adj.reshape(-1))


def adjstr_to_BN(row):
    # input: '0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    # output: [[0], [1, 1], [2, 0, 0], [3, 0, 0, 0], [4, 0, 1, 0, 0], [5, 1, 1, 0, 0, 0], [6, 0, 1, 0, 0, 1, 0], [7, 0, 0, 0, 1, 1, 1, 0]]
    matrix = np.array([int(x) for x in row.split()]).reshape(8, 8)
    res = [[0]]
    for i in range(1, 8):
        cur = [i] + matrix[:i, i].tolist()
        res.append(cur)
    return res


def decode_from_latent_space(
        latent_points, model, decode_attempts=500, n_nodes='variable', return_igraph=False,
        data_type='ENAS'):
    # decode points from the VAE model's latent space multiple attempts
    # and return the most common decoded graphs
    if n_nodes != 'variable':
        check_n_nodes = True  # check whether the decoded graphs have exactly n nodes
    else:
        check_n_nodes = False
    decoded_arcs = []  # a list of lists of igraphs
    pbar = tqdm(range(decode_attempts))
    for i in pbar:
        current_decoded_arcs = model.decode(latent_points)
        decoded_arcs.append(current_decoded_arcs)
        pbar.set_description("Decoding attempts {}/{}".format(i, decode_attempts))

    # We see which ones are decoded to be valid architectures
    valid_arcs = []  # a list of lists of strings
    if return_igraph:
        str2igraph = {}  # map strings to igraphs
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_arcs.append([])
        for j in range(decode_attempts):
            arc = decoded_arcs[j][i]  # arc is an igraph
            if data_type == 'ENAS':
                if is_valid_ENAS(arc, model.START_TYPE, model.END_TYPE):
                    if not check_n_nodes or check_n_nodes and arc.vcount() == n_nodes:
                        cur = decode_igraph_to_ENAS(arc)  # a flat ENAS string
                        if return_igraph:
                            str2igraph[cur] = arc
                        valid_arcs[i].append(cur)
            elif data_type == 'BN':
                if is_valid_BN(arc, model.START_TYPE, model.END_TYPE, nvt=model.nvt):
                    cur = decode_igraph_to_BN_adj(arc)  # a flat BN adjacency matrix string
                    if return_igraph:
                        str2igraph[cur] = arc
                    valid_arcs[i].append(cur)
        pbar.set_description("Check validity for {}/{}".format(i, latent_points.shape[0]))

    # select the most common decoding as the final architecture
    final_arcs = []  # a list of lists of strings
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_curs = valid_arcs[i]
        aux = collections.Counter(valid_curs)
        if len(aux) > 0:
            arc, num_arc = list(aux.items())[np.argmax(aux.values())]
        else:
            arc = None
            num_arc = 0
        final_arcs.append(arc)
        pbar.set_description("Latent point {}'s most common decoding ratio: {}/{}".format(
            i, num_arc, len(valid_curs)))

    if return_igraph:
        final_arcs_igraph = [str2igraph[x] if x is not None else None for x in final_arcs]
        return final_arcs_igraph, final_arcs
    return final_arcs


'''Network visualization'''


# plot_DAG(g_recon[0], args.res_dir, name1, data_type=args.data_type)
def plot_DAG(g, res_dir, name, backbone=False, data_type='ENAS', pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name + '.png')
    if pdf:
        file_name = os.path.join(res_dir, name + '.pdf')
    if data_type == 'ENAS':
        draw_network(g, file_name, backbone)
    elif data_type == 'ENAS_FULL':
        draw_network(g, file_name, backbone)
    elif data_type == 'BN':
        draw_BN(g, file_name)
    elif data_type == 'BN_FULL':
        draw_BN(g, file_name)
    elif data_type == 'NASBench_101':
        draw_NASBench(g, file_name, backbone=False)
    elif data_type == 'ENAS_MS':
        draw_network(g, file_name, backbone)
    elif data_type == 'ENAS_to_NASBench':
        draw_network(g, file_name, backbone=False)
    elif data_type == 'cora':
        draw_network(g, file_name, backbone=False)
    return file_name


def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx - 1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def draw_NASBench(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node_NASBench(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx - 1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 1:
        label = 'input'
        color = 'skyblue'
    elif label == 2:
        label = 'output'
        color = 'pink'
    elif label == 3:
        label = 'conv3'
        color = 'yellow'
    elif label == 4:
        label = 'sep3'
        color = 'orange'
    elif label == 5:
        label = 'conv5'
        color = 'greenyellow'
    elif label == 6:
        label = 'sep5'
        color = 'seagreen3'
    elif label == 7:
        label = 'avg3'
        color = 'azure'
    elif label == 8:
        label = 'max3'
        color = 'beige'
    elif label == 0:
        label = 'dm'
        color = 'red'
    else:
        label = ''
        color = 'aliceblue'

    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

def add_node_NASBench(graph, node_id, label, shape='box', style='filled'):
    if label == 1:
        label = 'input'
        color = 'skyblue'
    elif label == 2:
        label = 'output'
        color = 'pink'
    elif label == 3:
        label = 'conv1'
        color = 'yellow'
    elif label == 4:
        label = 'conv3'
        color = 'orange'
    elif label == 5:
        label = 'max3'
        color = 'greenyellow'
    # elif label == 6:
    #     label = 'sep5'
    #     color = 'seagreen3'
    # elif label == 7:
    #     label = 'avg3'
    #     color = 'azure'
    # elif label == 8:
    #     label = 'max3'
    #     color = 'beige'
    else:
        label = ''
        color = 'aliceblue'
    # label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
        node_id, label=label, color='black', fillcolor=color,
        shape=shape, style=style, fontsize=24)


def draw_BN(g, path):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    label_dict = dict(zip(range(2, 10), 'ASTLBEXD'))
    pos_dict = dict(zip(range(2, 10), ['0, 3!', '2.75, 3!', '0, 2!', '2, 2!', '3.5, 1!', '1.5, 1!', '1.5, 0!', '3.5, 0!']))

    def add_node(graph, node_id, label, shape='circle', style='filled'):
        if label in {0, 1}:
            return
        else:
            label, pos = label_dict[label], pos_dict[label]
        graph.add_node(
            node_id, label=label, color='black', fillcolor='white',
            shape=shape, style=style, pos=pos, fontsize=27,
        )
        return

    if g is None:
        graph.add_node(
            0, label='invalid', color='black', fillcolor='white',
            shape='box', style='filled',
        )
        graph.layout(prog='dot')
        graph.draw(path)
        return

    for idx in range(1, g.vcount() - 1):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(1, g.vcount() - 1):
        for node in g.get_adjlist(igraph.IN)[idx]:
            # if node != g.vcount()-1 and node != 0:  # we don't draw input/output nodes for BN
            node_type = g.vs[node]['type']
            if node_type != 0 and node_type != 1:  # we don't draw input/output nodes for BN
                graph.add_edge(node, idx)

    graph.layout()
    graph.draw(path)
    return path


'''Validity and novelty functions'''


def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True


def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
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


def is_valid_ENAS(g, START_TYPE=0, END_TYPE=1):
    # first need to be a valid DAG computation graph
    res = is_valid_DAG(g, START_TYPE, END_TYPE)

    # in addition, node i must connect to node i+1
    # for i in range(g.vcount() - 2):
    #     res = res and g.are_connected(i, i + 1)
    #     if not res:
    #         return res
    # the output node n must not have edges other than from n-1
    # res = res and (g.vs[g.vcount() - 1].indegree() == 1)
    return res


def is_valid_BN(g, START_TYPE=0, END_TYPE=1, nvt=10):
    # nvt: number of vertex types in this BN
    # first need to be a DAG
    res = g.is_dag()
    # check whether start and end types only appear once
    # BN nodes need not be connected
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
    res = res and n_start == 1 and n_end == 1
    # in addition, every type must appear exactly once
    res = res and (len(set(g.vs['type'])) == nvt) and g.vcount() == nvt
    return res


'''Other util functions'''


def combine_figs_horizontally(names, new_name):
    images = list(map(Image.open, names))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(new_name)


class custom_DataParallel(nn.parallel.DataParallel):
    # define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G) - 1:
                scattered_G.append((mini_batch,))
                mini_batch = []
        return tuple(scattered_G), tuple([{}] * len(scattered_G))
