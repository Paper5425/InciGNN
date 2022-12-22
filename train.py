from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
import tqdm
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr
import igraph
from random import shuffle
import matplotlib
import scipy.stats as stats
from scipy.stats import pearsonr

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN
import logging
import shutil

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='ENAS_FULL', help='ENAS_FULL: ENAS-format CNN structures; BN: Bayesian networks, NASBench_101, ENAS_to_NASBench')
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name: final_structures6, nas_bench_101')
parser.add_argument('--nvt', type=int, default=6, help='number of different node types, 6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='_wo_gnn', help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=100, metavar='N', help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N', help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False, help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False, help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=False, help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False, help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False, help='if True, use a smaller version of train set')
parser.add_argument('--temp', type=float, default=1.0, metavar='S', help='tau(temperature) (default: 1.0)')
parser.add_argument('--link-prediction', default=False, help='if True, mask the link randomly')
parser.add_argument('--add-noise', default=False, help='if True, add noise for link prediction')

# model settings
parser.add_argument('--model', default='DVAE_INCI_WO_GNN', help='model to use: DVAE, SVAE, DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--load-latest-model', action='store_true', default=False, help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=64, metavar='N'  , help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=32, metavar='N', help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False, help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False, help='whether to train a performance predictor from latent encodings and a VAE at the same time')

# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=32, metavar='N', help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False, help='use all available GPUs')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save', type=str, default='logs')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.save_appendix))
args.save = args.res_dir

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

# create log file
# if not os.path.isfile(os.path.join(args.save, 'log.txt')):
#     file = open(os.path.join(args.save, 'log.txt'), 'wb')
#     file.close()

# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)

# otherwise process the raw data and save to .pkl
else:
    # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
    if args.model.startswith('DVAE'):
        input_fmt = 'igraph'
    elif args.model.startswith('SVAE'):
        input_fmt = 'string'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt, fmt=input_fmt)
    elif args.data_type == 'ENAS_MAT':
        train_data, test_data, graph_args = load_ENAS_matrix(args.data_name, n_types=args.nvt, fmt=input_fmt)
    elif args.data_type == 'ENAS_MS':
        train_data, test_data, graph_args = load_ENAS_matrix_MS(args.data_name, n_types=args.nvt, fmt=input_fmt)
    elif args.data_type == 'NASBench_101':
        train_data, test_data, graph_args = load_NASBench_matrix(args.data_name, n_types=args.nvt, fmt=input_fmt, link_pred=args.link_prediction, add_noise=args.add_noise)
    elif args.data_type == 'ENAS_FULL':
        train_data, test_data, graph_args = load_ENAS_matrix_Full(args.data_name, n_types=args.nvt, fmt=input_fmt, link_pred=args.link_prediction, add_noise=args.add_noise)
    # elif args.data_type == 'BN':
    #     train_data, test_data, graph_args = load_BN_matrix_Full(args.data_name, n_types=args.nvt, fmt=input_fmt)
    elif args.data_type == 'ENAS_to_NASBench':
        train_data, test_data, graph_args = load_ENAS_to_NASBench(args.data_name, n_types=args.nvt, fmt=input_fmt)
    elif args.data_type == 'WikiCS':
        train_data, test_data, graph_args = load_WikiCS_subs(args.data_name)
    elif args.data_type == 'BN':
        train_data, test_data, graph_args = load_BN_graphs(args.data_name, n_types=args.nvt, fmt=input_fmt)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)


# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and
               not f.startswith('train_graph') and not f.startswith('test_graph') and
               not f.endswith('.pth') and
               not f.startswith('log')]
for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

if not args.keep_old:
    # backup current .py files
    shutil.copy('train.py', args.res_dir)
    shutil.copy('models.py', args.res_dir)
    shutil.copy('util.py', args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]


'''Prepare the model'''
# model
model = eval(args.model)(
    graph_args.max_n,
    graph_args.max_n_eg,
    graph_args.num_vertex_type,
    graph_args.START_TYPE,
    graph_args.END_TYPE,
    hidden_dim=args.hs,
    z_dim=args.nz,
)


class RankLoss(torch.nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, pred_1, pred_2, y_1, y_2):
        x = (pred_1 - pred_2) * torch.sign(y_1 - y_2)
        loss = self.phy(x)
        return torch.sum(loss)

    def phy(self, x):
        y = torch.log(1 + torch.exp(-x))
        return y


if args.predictor:
    predictor = nn.Sequential(
        nn.Linear(192, args.hs),
        nn.Tanh(),
        nn.Linear(args.hs, 1)
    )
    model.predictor = predictor
    model.rankloss = RankLoss()
    model.mseloss = nn.MSELoss(reduction='sum')
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))

if args.link_prediction:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch)))
        # load_module_state(optimizer, os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)))
        # load_module_state(scheduler, os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch)))
    else:
        print('Please set model checkpoint!')

else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch)))
        load_module_state(optimizer, os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)))
        load_module_state(scheduler, os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch)))

# plot sample train/test graphs
if not os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or args.reprocess:
    if not args.keep_old:
        for data in ['train_data', 'test_data']:
            G = [g[-1] for g in eval(data)[:10]]
            if args.model.startswith('SVAE'):
                G = [g.to(device) for g in G]
                G = model._collate_fn(G)
                G = model.construct_igraph(G[:, :, :model.nvt], G[:, :, model.nvt:], False)
            for i, g in enumerate(G):
                name = '{}_graph_id{}'.format(data[:-5], i)
                plot_DAG(g, args.res_dir, name, data_type=args.data_type)

temp_min = 0.3
ANNEAL_RATE = 0.00003

'''Define some train/test functions'''


def train(epoch):
    model.train()
    train_loss = 0
    adj_ls = 0
    inci_ls = 0
    ver_ls = 0
    inci_T_ls = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    pbar = tqdm.tqdm(train_data)

    acc_batch = []
    adj_batch = []
    adj_N_batch = []
    nodes_batch = []
    ops_batch = []
    inci_T_batch = []
    inci_H_batch = []
    inci_D_batch = []
    edges_batch = []
    edges_T_batch = []
    weight_batch = []
    weight_T_batch = []
    weight_H_batch = []
    gh_batch = []

    # y, adj_mat, adj_mat_N, ops, inci_lb_T, inci_lb_H, inci_mat_T, edge_t, weight_tensor, weight_tensor_T, weight_tensor_H, g

    # y, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g
    #    adj_mat, ops,       ops,   ops, inci_lb_T, inci_lb_H, ops,  inci_lb_T, inci_lb_H, weight_tensor, weight_tensor_T, weight_tensor_H, g

    for i, (ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g) in enumerate(pbar):
        # if args.model.startswith('SVAE'):  # for SVAE, g is tensor
        #     g = g.to(device)
        acc_batch.append(ac.to(device))
        adj_batch.append(ad.to(device))
        adj_N_batch.append(ad_N.to(device))
        nodes_batch.append(nd.to(device))
        ops_batch.append(op.to(device))
        inci_T_batch.append(inc_T.to(device))
        inci_H_batch.append(inc_H.to(device))
        inci_D_batch.append(inc_D.to(device))
        edges_batch.append(eds.to(device))
        edges_T_batch.append(eds_t.to(device))
        weight_batch.append(wt.to(device))
        weight_T_batch.append(wt_T.to(device))
        weight_H_batch.append(wt_N.to(device))
        gh_batch.append(g)

        if len(adj_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = (
            acc_batch, adj_batch, adj_N_batch, nodes_batch, ops_batch, inci_T_batch, inci_H_batch, inci_D_batch,
            edges_batch, edges_T_batch, weight_batch, weight_T_batch, weight_H_batch, gh_batch)
            # g_batch = model._collate_fn(g_batch)
            if args.all_gpus:  # does not support predictor yet
                loss = net(g_batch).sum()

                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item() / len(g_batch)))
                adj_loss, inci_loss, kld = 0, 0, 0
            else:
                mean, logvar, sampled_z = model.encode(g_batch)
                loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = model.loss(mean, logvar, sampled_z, g_batch)
                if args.predictor:
                    y_batch = torch.FloatTensor(acc_batch).unsqueeze(1).to(device)
                    mu = torch.cat([mean[2].sum(dim=1), mean[1].sum(dim=1)], dim=-1)
                    # mu = torch.cat([mu, mean[2].sum(dim=1)], dim=-1)

                    y_pred = model.predictor(mu)
                    pred = model.mseloss(y_pred, y_batch)
                    #
                    # pred = 0
                    for i in range(y_pred.size(0) - 1):
                        for j in range(i+1, y_pred.size(0)):
                            pred = pred + model.rankloss(y_pred[i], y_pred[j], y_batch[i], y_batch[j])

                    loss = loss + pred
                    pbar.set_description(
                        'Epoch: %d, loss: %0.4f, ver: %0.4f, adj: %0.4f, inci: %0.4f, inci_T: %0.4f, kld: %0.4f, pred: %0.4f' %
                        (epoch, loss.item() / len(g_batch), ver_loss.item() / len(g_batch),
                         adj_loss.item() / len(g_batch), inci_loss.item() / len(g_batch),
                         inci_T_loss.item() / len(g_batch), kld.item() / len(g_batch), pred / len(g_batch)))
                else:
                    pbar.set_description(
                        'Epoch: %d, loss: %0.4f, ver: %0.4f, adj: %0.4f, inci: %0.4f, inci_T: %0.4f, kld: %0.4f' %
                        (epoch, loss.item() / len(g_batch), ver_loss.item() / len(g_batch),
                         adj_loss.item() / len(g_batch), inci_loss.item() / len(g_batch),
                         inci_T_loss.item() / len(g_batch), kld.item() / len(g_batch)))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

            # inci_loss.backward()

            train_loss += float(loss)
            ver_ls += float(ver_loss)
            adj_ls += float(adj_loss)
            inci_ls += float(inci_loss)
            inci_T_ls += float(inci_T_loss)
            kld_loss += float(kld)

            if args.predictor:
                pred_loss = pred_loss + float(pred)

            optimizer.step()
            acc_batch = []
            adj_batch = []
            adj_N_batch = []
            nodes_batch = []
            ops_batch = []
            inci_T_batch = []
            inci_H_batch = []
            inci_D_batch = []
            edges_batch = []
            edges_T_batch = []
            weight_batch = []
            weight_T_batch = []
            weight_H_batch = []
            gh_batch = []

    if args.predictor:
        logging.info("Epoch: %03d | Average loss: %.6f | Ver: %.6f | Adj: %.6f | Inci: %.6f | Inci_T: %.6f | KL: %.6f | Pred: %.6f |",
                     epoch, train_loss / len(train_data)
                     , ver_ls / len(train_data), adj_ls / len(train_data), inci_ls / len(train_data),
                     inci_T_ls / len(train_data), kld_loss / len(train_data), pred_loss / len(train_data))
        return train_loss, ver_ls, adj_ls, inci_ls, inci_T_ls, kld_loss, pred_loss


    logging.info("Epoch: %03d | Average loss: %.6f | Ver: %.6f | Adj: %.6f | Inci: %.6f | Inci_T: %.6f | KL: %.6f |",
                 epoch, train_loss / len(train_data)
                 , ver_ls / len(train_data), adj_ls / len(train_data), inci_ls / len(train_data),
                 inci_T_ls / len(train_data), kld_loss / len(train_data))

    return train_loss, ver_ls, adj_ls, inci_ls, inci_T_ls, kld_loss


def _to_graph(adj, ops):
    # matrix pruning
    adj = torch.triu(adj.squeeze(dim=0), 1).tolist()
    g = igraph.Graph.Adjacency(adj)
    for i, item in enumerate(ops.squeeze(dim=0).tolist()):
        g.vs[i]['type'] = item
    return g


def visualize_recon(epoch):
    model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality

    for i, g in enumerate(test_data[:10] + train_data[:10]):
        if args.model.startswith('SVAE'):
            g = g.to(device)
            g = model._collate_fn(g)
            g_recon = model.encode_decode(g)[0]
            g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)[0]
        elif args.model.startswith('DVAE'):
            # adj_pred, v_type_pred, inci_pred = model.encode_decode(g)
            g_recon = model.encode_decode(g)
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)

        # g_recon = _to_graph(adj_pred, v_type_pred)

        plot_DAG(g[-1], args.res_dir, name0, data_type=args.data_type)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon[0], args.res_dir, name1, data_type=args.data_type)


def test_during_train():
    # test recon accuracy
    model.eval()
    edge_error = 0
    vertex_error = 0
    kendall_tau = 0
    count = 0

    print('Accuracy testing begins...')

    pbar = tqdm.tqdm(test_data)

    acc_batch = []
    adj_batch = []
    adj_N_batch = []
    nodes_batch = []
    ops_batch = []
    inci_T_batch = []
    inci_H_batch = []
    inci_D_batch = []
    edges_batch = []
    edges_T_batch = []
    weight_batch = []
    weight_T_batch = []
    weight_H_batch = []
    gh_batch = []

    for i, (ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g) in enumerate(pbar):
        # if args.model.startswith('SVAE'):
        #     g = g.to(device)
        #
        acc_batch.append(ac.to(device))
        adj_batch.append(ad.to(device))
        adj_N_batch.append(ad_N.to(device))
        nodes_batch.append(nd.to(device))
        ops_batch.append(op.to(device))
        inci_T_batch.append(inc_T.to(device))
        inci_H_batch.append(inc_H.to(device))
        inci_D_batch.append(inc_D.to(device))
        edges_batch.append(eds.to(device))
        edges_T_batch.append(eds_t.to(device))
        weight_batch.append(wt.to(device))
        weight_T_batch.append(wt_T.to(device))
        weight_H_batch.append(wt_N.to(device))
        gh_batch.append(g)

        if len(adj_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g_batch = (acc_batch, adj_batch, adj_N_batch, nodes_batch, ops_batch, inci_T_batch, inci_H_batch, inci_D_batch,
                edges_batch, edges_T_batch, weight_batch, weight_T_batch, weight_H_batch, gh_batch)
            # g = model._collate_fn(g_batch)
            mean, _, sampled_z = model.encode(g_batch)
            vertex_pred, inci_pred = model.calculate_accuracy(sampled_z)

            inci_mat_T = torch.cat(inci_T_batch, dim=0)
            inci_mat_H = torch.cat(inci_H_batch, dim=0)
            vertex = torch.cat(ops_batch, dim=0)
            acc = torch.cat(acc_batch, dim=0)

            inci = inci_mat_T * -1 + inci_mat_H

            edge_error = edge_error + (inci != inci_pred.permute(0, 2, 1)).float().sum().tolist() # torch.abs((inci.permute(0, 2, 1) + inci_pred).float()).sum().tolist()
            vertex_error = vertex_error + (vertex_pred != vertex).float().sum().tolist()

            # _, inci_pred_T, inci_pred_H = model._decode(sampled_z)

            mean_vert = mean[0]
            mean_edge_T = mean[1]
            mean_edge_H = mean[2]

            # vet = torch.bmm(mean_edge_T, mean_vert.permute(0, 2, 1))
            # veh = torch.bmm(mean_edge_H, mean_vert.permute(0, 2, 1))

            inci_lb_T = torch.cat(inci_T_batch, dim=0)
            inci_lb_H = torch.cat(inci_H_batch, dim=0)

            acc_pred = model.pred_acc_2nd(mean_vert, inci_lb_T, inci_lb_H)

            # acc_pred = model.pred_acc(vet, veh, inci_lb_T, inci_lb_H)

            # acc_pred = model.pred_acc(inci_pred_T, inci_pred_H, inci_mat_T, inci_mat_H)
            # tau, p_value = stats.kendalltau(acc_pred.squeeze(dim=-1).cpu().tolist(), acc.cpu().tolist())
            tau, _ = pearsonr(acc_pred.squeeze(dim=-1).cpu().tolist(), acc.cpu().tolist())

            kendall_tau = kendall_tau + tau
            count = count + 1

            acc_batch = []
            adj_batch = []
            adj_N_batch = []
            nodes_batch = []
            ops_batch = []
            inci_T_batch = []
            inci_H_batch = []
            inci_D_batch = []
            edges_batch = []
            edges_T_batch = []
            weight_batch = []
            weight_T_batch = []
            weight_H_batch = []
            gh_batch = []

    vertex_error = vertex_error / (len(test_data) * 8)
    edge_error = edge_error / (len(test_data) * 8 * 28)
    kendall_tau = kendall_tau / float(count)

    logging.info('Test average Vertex error: {0}, Edge error: {1:.4f}, Kendall_Tau: {2:.4f}'.format(vertex_error, edge_error, kendall_tau))
    return vertex_error, edge_error, kendall_tau


def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    count = 0
    count_k = 0
    kendall_tau = 0
    print('Testing begins...')

    pbar = tqdm.tqdm(test_data)

    acc_batch = []
    adj_batch = []
    adj_N_batch = []
    nodes_batch = []
    ops_batch = []
    inci_T_batch = []
    inci_H_batch = []
    inci_D_batch = []
    edges_batch = []
    edges_T_batch = []
    weight_batch = []
    weight_T_batch = []
    weight_H_batch = []
    gh_batch = []

    for i, (ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g) in enumerate(pbar):
        # if args.model.startswith('SVAE'):
        #     g = g.to(device)
        #
        acc_batch.append(ac.to(device))
        adj_batch.append(ad.to(device))
        adj_N_batch.append(ad_N.to(device))
        nodes_batch.append(nd.to(device))
        ops_batch.append(op.to(device))
        inci_T_batch.append(inc_T.to(device))
        inci_H_batch.append(inc_H.to(device))
        inci_D_batch.append(inc_D.to(device))
        edges_batch.append(eds.to(device))
        edges_T_batch.append(eds_t.to(device))
        weight_batch.append(wt.to(device))
        weight_T_batch.append(wt_T.to(device))
        weight_H_batch.append(wt_N.to(device))
        gh_batch.append(g)

        if len(adj_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g_batch = (acc_batch, adj_batch, adj_N_batch, nodes_batch, ops_batch, inci_T_batch, inci_H_batch, inci_D_batch,
                       edges_batch, edges_T_batch, weight_batch, weight_T_batch, weight_H_batch, gh_batch)
            # g = model._collate_fn(g_batch)
            mean, logvar, sampled_z = model.encode(g_batch)
            loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = model.loss(mean, logvar, sampled_z, g_batch)
            # _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(loss.item() / len(g_batch)))
            Nll += loss.item()
            if args.predictor:
                y_batch = torch.FloatTensor(acc_batch).unsqueeze(1).to(device)
                mu = torch.cat([mean[2].sum(dim=1), mean[1].sum(dim=1)], dim=-1)
                # mu = torch.cat([mu, mean[2].sum(dim=1)], dim=-1)

                y_pred = model.predictor(mu)
                pred = model.mseloss(y_pred, y_batch)

                for i in range(y_pred.size(0) - 1):
                    for j in range(i + 1, y_pred.size(0)):
                        pred = pred + model.rankloss(y_pred[i], y_pred[j], y_batch[i], y_batch[j])

                pred_loss = pred_loss + pred.item()
            # construct igraph g from tensor g to check recon quality
            if args.model.startswith('SVAE'):
                g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)

            for _ in range(decode_times):
                g_recon = model.decode(sampled_z)
                for i, (g0, g1) in enumerate(zip(g_batch[-1], g_recon)):
                    # name0 = 'graph_test{}_id{}_ori'.format(count, i)
                    # plot_DAG(g0, args.res_dir, name0, data_type=args.data_type)
                    # name1 = 'graph_test{}_id{}_rec'.format(count, i)
                    # plot_DAG(g1, args.res_dir, name1, data_type=args.data_type)

                    if is_same_DAG(g0, g1):
                        n_perfect = n_perfect + 1

                    count = count + 1

            inci_mat_T = torch.cat(inci_T_batch, dim=0)
            inci_mat_H = torch.cat(inci_H_batch, dim=0)
            acc = torch.cat(acc_batch, dim=0)
            # _, inci_pred_T, inci_pred_H = model._decode(sampled_z)
            # acc_pred = model.pred_acc(inci_pred_T, inci_pred_H, inci_mat_T, inci_mat_H)
            # tau, p_value = stats.kendalltau(acc_pred.squeeze(dim=-1).cpu().tolist(), acc.cpu().tolist())

            mean_vert = mean[0]
            mean_edge_T = mean[1]
            mean_edge_H = mean[2]

            # vet = torch.bmm(mean_edge_T, mean_vert.permute(0, 2, 1))
            # veh = torch.bmm(mean_edge_H, mean_vert.permute(0, 2, 1))

            inci_lb_T = torch.cat(inci_T_batch, dim=0)
            inci_lb_H = torch.cat(inci_H_batch, dim=0)
            acc_pred = model.pred_acc_2nd(mean_vert, inci_lb_T, inci_lb_H)
            # acc_pred = model.pred_acc(vet, veh, inci_lb_T, inci_lb_H)

            tau, _ = pearsonr(acc_pred.squeeze(dim=-1).cpu().tolist(), acc.cpu().tolist())
            kendall_tau = kendall_tau + tau
            count_k = count_k + 1

            acc_batch = []
            adj_batch = []
            adj_N_batch = []
            nodes_batch = []
            ops_batch = []
            inci_T_batch = []
            inci_H_batch = []
            inci_D_batch = []
            edges_batch = []
            edges_T_batch = []
            weight_batch = []
            weight_T_batch = []
            weight_H_batch = []
            gh_batch = []

    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / count
    kendall_tau = kendall_tau / float(count_k)
    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}'.format(
            Nll, acc, pred_rmse))
        return Nll, acc, pred_rmse
    else:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, kendall_tau: {2:.4f}'.format(Nll, acc, kendall_tau))
        return Nll, acc, kendall_tau

# Link prediction
def link_pred():
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    f1_collect = 0
    count = 0
    ktau_coef = 0
    k_count = 0

    print('Link prediction begins...')

    pbar = tqdm.tqdm(test_data)

    acc_batch = []
    adj_batch = []
    adj_N_batch = []
    nodes_batch = []
    ops_batch = []
    inci_T_batch = []
    inci_H_batch = []
    inci_D_batch = []
    edges_batch = []
    edges_T_batch = []
    weight_batch = []
    weight_T_batch = []
    weight_H_batch = []
    gh_batch = []

    for i, (ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g) in enumerate(pbar):
        # if args.model.startswith('SVAE'):
        #     g = g.to(device)
        #
        acc_batch.append(ac)
        adj_batch.append(ad.to(device))
        adj_N_batch.append(ad_N.to(device))
        nodes_batch.append(nd.to(device))
        ops_batch.append(op.to(device))
        inci_T_batch.append(inc_T.to(device))
        inci_H_batch.append(inc_H.to(device))
        inci_D_batch.append(inc_D.to(device))
        edges_batch.append(eds.to(device))
        edges_T_batch.append(eds_t.to(device))
        weight_batch.append(wt.to(device))
        weight_T_batch.append(wt_T.to(device))
        weight_H_batch.append(wt_N.to(device))
        gh_batch.append(g)

        if len(adj_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g_batch = (acc_batch, adj_batch, adj_N_batch, nodes_batch, ops_batch, inci_T_batch, inci_H_batch, inci_D_batch,
                       edges_batch, edges_T_batch, weight_batch, weight_T_batch, weight_H_batch, gh_batch)
            # g = model._collate_fn(g_batch)
            mean, logvar, sampled_z = model.encode(g_batch)
            loss, ver_loss, adj_loss, inci_loss, inci_T_loss, kld = model.loss(mean, logvar, sampled_z, g_batch)
            # _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(loss.item() / len(g_batch)))
            Nll += loss.item()
            if args.predictor:
                y_batch = torch.FloatTensor(acc_batch).unsqueeze(1).to(device)
                mu = torch.cat([mean[2].sum(dim=1), mean[1].sum(dim=1)], dim=-1)
                # mu = torch.cat([mu, mean[2].sum(dim=1)], dim=-1)

                y_pred = model.predictor(mu)
                pred = model.mseloss(y_pred, y_batch)

                for i in range(y_pred.size(0) - 1):
                    for j in range(i + 1, y_pred.size(0)):
                        pred = pred + model.rankloss(y_pred[i], y_pred[j], y_batch[i], y_batch[j])

                pred_loss += pred.item()

                # ktau, _ = stats.kendalltau(y_pred.cpu().tolist(), y_batch.cpu().tolist())



                ktau, _ = stats.pearsonr(y_pred.view(-1).cpu().tolist(), y_batch.view(-1).cpu().tolist())
                ktau_coef = ktau_coef + ktau
                k_count = k_count + 1

            # construct igraph g from tensor g to check recon quality
            if args.model.startswith('SVAE'):
                g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)

            # for _ in range(encode_times):
            #     # z = model.reparameterize(mean, logvar)
            for _ in range(decode_times):
                acc, f1 = model.link_predictor(sampled_z, g_batch)

                n_perfect = acc + n_perfect
                f1_collect = f1_collect + f1
                count = count + 1

                # g_recon = model.decode(sampled_z)
                # n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g_batch[-1], g_recon))

            acc_batch = []
            adj_batch = []
            adj_N_batch = []
            nodes_batch = []
            ops_batch = []
            inci_T_batch = []
            inci_H_batch = []
            inci_D_batch = []
            edges_batch = []
            edges_T_batch = []
            weight_batch = []
            weight_T_batch = []
            weight_H_batch = []
            gh_batch = []

    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc_avg = n_perfect / count
    f1_avg = f1_collect / count
    ktau_avg = ktau_coef / k_count

    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}, ktau: {3:.4f}'.format(
            Nll, acc, pred_rmse, ktau_avg))
        return Nll, acc, pred_rmse
    else:
        print('Edge prediction accuracy: {0}, the F1 score: {1:.4f}'.format(acc_avg, f1_avg))
        return Nll, acc


def prior_validity(scale_to_train_range=True):
    if scale_to_train_range:
        Z_train, Y_train = extract_latent(test_data)
        # z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        #
        # z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    n_latent_points = 1000
    decode_times = 1
    n_valid = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    G_train = [g for ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g in train_data]

    if args.model.startswith('SVAE'):
        G_train = [g.to(device) for g in G_train]
        G_train = model._collate_fn(G_train)
        G_train = model.construct_igraph(G_train[:, :, :model.nvt], G_train[:, :, model.nvt:], False)
    pbar = tqdm.tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == args.infer_batch_size or i == n_latent_points - 1:
            # z = torch.randn(cnt, model.nz).to(model.get_device())
            z_vertex = torch.randn(cnt, model.max_n_vertex, model.nz + model.hs * 0).to(model.get_device())
            z_edge_T = torch.randn(cnt, model.max_n_edge, model.nz + model.hs * 0).to(model.get_device())
            z_edge_H = torch.randn(cnt, model.max_n_edge, model.nz + model.hs * 0).to(model.get_device())

            if scale_to_train_range:
                z_vertex = z_vertex * Z_train[3] + Z_train[0]
                z_edge_T = z_edge_T * Z_train[4] + Z_train[1]
                z_edge_H = z_edge_H * Z_train[5] + Z_train[2]

            z = (z_vertex, z_edge_T, z_edge_H)
            # z = (z_vertex.unsqueeze(dim=0), z_edge_T.unsqueeze(dim=0), z_edge_H.unsqueeze(dim=0))
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                if args.data_type == 'ENAS_FULL':
                    for g in g_batch:
                        if is_valid_ENAS(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
                        # else:
                        #     print('*' * 20)
                        #     print(g)
                        #     for v in range(g.vcount()):
                        #         print(g.vs[v])
                        #     print('#' * 20)


                elif args.data_type == 'BN':
                    for g in g_batch:
                        if is_valid_BN(g, graph_args.START_TYPE, graph_args.END_TYPE):
                            n_valid += 1
                            G_valid.append(g)
            cnt = 0

    r_valid = n_valid / (n_latent_points * decode_times)
    print('Ratio of valid decodings from the prior: {:.4f}'.format(r_valid))

    G_valid_str = [decode_igraph_to_ENAS(g) for g in G_valid]
    r_unique = len(set(G_valid_str)) / len(G_valid_str) if len(G_valid_str) != 0 else 0.0
    print('Ratio of unique decodings from the prior: {:.4f}'.format(r_unique))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))

    return r_valid, r_unique, r_novel


def extract_latent(data):
    model.eval()
    mean_vert = []
    mean_edge_T = []
    mean_edge_H = []

    logvar_vert = []
    logvar_edge_T = []
    logvar_edge_H = []

    Z = []

    acc_batch = []
    adj_batch = []
    adj_N_batch = []
    nodes_batch = []
    ops_batch = []
    inci_T_batch = []
    inci_H_batch = []
    inci_D_batch = []
    edges_batch = []
    edges_T_batch = []
    weight_batch = []
    weight_T_batch = []
    weight_H_batch = []
    gh_batch = []

    for i, (ac, ad, ad_N, nd, op, inc_T, inc_H, inc_D, eds, eds_t, wt, wt_T, wt_N, g) in enumerate(tqdm.tqdm(data)):
        # acc_batch.append(ac)
        # adj_batch.append(ad.to(device))
        # adj_N_batch.append(ad_N.to(device))
        # nodes_batch.append(nd.to(device))
        ops_batch.append(op.to(device))
        inci_T_batch.append(inc_T.to(device))
        inci_H_batch.append(inc_H.to(device))
        # inci_D_batch.append(inc_D.to(device))
        # edges_batch.append(eds.to(device))
        # edges_T_batch.append(eds_t.to(device))
        # weight_batch.append(wt.to(device))
        # weight_T_batch.append(wt_T.to(device))
        # weight_H_batch.append(wt_N.to(device))
        # gh_batch.append(g)

        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DVAE'):
            # copy igraph
            # otherwise original igraphs will save the H states and consume more GPU memory
            # g_ = g.copy()
            a = 0

        if len(adj_batch) == args.infer_batch_size or i == len(data) - 1:
            # g_batch = model._collate_fn(g_batch)
            g_batch = (acc_batch, adj_batch, adj_N_batch, nodes_batch, ops_batch, inci_T_batch, inci_H_batch, inci_D_batch,
                       edges_batch, edges_T_batch, weight_batch, weight_T_batch, weight_H_batch, gh_batch)
            _, _, z_pred = model.encode(g_batch)

            Z.append(z_pred)

            # mean_vert.append(mu[0])
            # mean_edge_T.append(mu[1])
            # mean_edge_H.append(mu[2])
            #
            # logvar_vert.append(logvar[0])
            # logvar_edge_T.append(logvar[1])
            # logvar_edge_H.append(logvar[2])

            acc_batch = []
            adj_batch = []
            adj_N_batch = []
            nodes_batch = []
            ops_batch = []
            inci_T_batch = []
            inci_H_batch = []
            inci_D_batch = []
            edges_batch = []
            edges_T_batch = []
            weight_batch = []
            weight_T_batch = []
            weight_H_batch = []
            gh_batch = []

    mu_v = z_pred[0].mean(dim=0)
    mu_e_T = z_pred[1].mean(dim=0)
    mu_e_H = z_pred[2].mean(dim=0)

    lv_v = z_pred[0].var(dim=0)
    lv_e_T = z_pred[1].var(dim=0)
    lv_e_H = z_pred[2].var(dim=0)


    return [mu_v, mu_e_T, mu_e_H, lv_v, lv_e_T, lv_e_H], np.array(acc_batch)


'''Extract latent representations Z'''


def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name + '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)

    Z_train = [Z_train[0].cpu().detach().numpy(),
               Z_train[1].cpu().detach().numpy(),
               Z_train[2].cpu().detach().numpy()]

    Z_test = [Z_test[0].cpu().detach().numpy(),
              Z_test[1].cpu().detach().numpy(),
              Z_test[2].cpu().detach().numpy()]

    scipy.io.savemat(latent_mat_name,
                     mdict={
                         'Z_train': Z_train,
                         'Z_test': Z_test,
                         'Y_train': Y_train,
                         'Y_test': Y_test
                     }
                     )


def interpolation_exp(epoch, num=5):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir)
    if args.data_type == 'BN':
        eva = Eval_BN(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    cnt = 0
    for i in range(0, len(test_data), 2):
        cnt += 1
        (g0, _), (g1, _) = test_data[i], test_data[i + 1]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g1 = g1.to(device)
            g0 = model._collate_fn([g0])
            g1 = model._collate_fn([g1])
        z0, _ = model.encode(g0)
        z1, _ = model.encode(g1)
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0 - z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)
        # decode many times and select the most common one
        G, G_str = decode_from_latent_space(Z, model, return_igraph=True,
                                            data_type=args.data_type)
        names = []
        scores = []
        for j in range(0, interpolate_number + 1):
            namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True,
                             data_type=args.data_type)
            names.append(namej)
            if args.data_type == 'BN':
                scorej = eva.eval(G_str[j])
                scores.append(scorej)
        fig = plt.figure(figsize=(120, 20))
        for j, namej in enumerate(names):
            imgj = mpimg.imread(namej)
            fig.add_subplot(1, interpolate_number + 1, j + 1)
            plt.imshow(imgj)
            if args.data_type == 'BN':
                plt.title('{}'.format(scores[j]), fontsize=40)
            plt.axis('off')
        plt.savefig(os.path.join(args.res_dir,
                                 args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
                                     args.model, epoch, i)), bbox_inches='tight')
        '''
        # draw figures with the same height
        new_name = os.path.join(args.res_dir, args.data_name + 
                                '_{}_interpolate_exp_ensemble_{}.pdf'.format(args.model, i))
        combine_figs_horizontally(names, new_name)
        '''
        if cnt == num:
            break


def interpolation_exp2(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments between flat-net and dense-net')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation2')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir)
    interpolate_number = 10
    model.eval()
    n = graph_args.max_n
    g0 = [[1] + [0] * (i - 1) for i in range(1, n - 1)]  # this is flat-net
    g1 = [[1] + [1] * (i - 1) for i in range(1, n - 1)]  # this is dense-net

    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g1, _ = decode_ENAS_to_tensor(str(g1), n_types=6)
        g0 = g0.to(device)
        g1 = g1.to(device)
        g0 = model._collate_fn([g0])
        g1 = model._collate_fn([g1])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
        g1, _ = decode_ENAS_to_igraph(str(g1))
    z0, _ = model.encode(g0)
    z1, _ = model.encode(g1)
    print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0 - z1)))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        zj = z0 + (z1 - z0) / interpolate_number * j
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True,
                         data_type=args.data_type)
        names.append(namej)
    fig = plt.figure(figsize=(120, 20))
    for j, namej in enumerate(names):
        imgj = mpimg.imread(namej)
        fig.add_subplot(1, interpolate_number + 1, j + 1)
        plt.imshow(imgj)
        plt.axis('off')
    plt.savefig(os.path.join(args.res_dir,
                             args.data_name + '_{}_interpolate_exp2_ensemble_epoch{}.pdf'.format(
                                 args.model, epoch)), bbox_inches='tight')


def interpolation_exp3(epoch):
    if args.data_type != 'ENAS':
        return
    print('Interpolation experiments around a great circle')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation3')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir)
    interpolate_number = 36
    model.eval()
    n = graph_args.max_n
    g0 = [[1] + [0] * (i - 1) for i in range(1, n - 1)]  # this is flat-net
    if args.model.startswith('SVAE'):
        g0, _ = decode_ENAS_to_tensor(str(g0), n_types=6)
        g0 = g0.to(device)
        g0 = model._collate_fn([g0])
    elif args.model.startswith('DVAE'):
        g0, _ = decode_ENAS_to_igraph(str(g0))
    z0, _ = model.encode(g0)
    norm0 = torch.norm(z0)
    z1 = torch.ones_like(z0)
    # there are infinite possible directions that are orthogonal to z0,
    # we just randomly pick one from a finite set
    dim_to_change = random.randint(0, z0.shape[1] - 1)  # this to get different great circles
    print(dim_to_change)
    z1[0, dim_to_change] = -(z0[0, :].sum() - z0[0, dim_to_change]) / z0[0, dim_to_change]
    z1 = z1 / torch.norm(z1) * norm0
    print('z0: ', z0, 'z1: ', z1, 'dot product: ', (z0 * z1).sum().item())
    print('norm of z0: {}, norm of z1: {}'.format(norm0, torch.norm(z1)))
    print('distance between z0 and z1: {}'.format(torch.norm(z0 - z1)))
    omega = torch.acos(torch.dot(z0.flatten(), z1.flatten()))
    print('angle between z0 and z1: {}'.format(omega))
    Z = []  # to store all the interpolation points
    for j in range(0, interpolate_number + 1):
        theta = 2 * math.pi / interpolate_number * j
        zj = z0 * np.cos(theta) + z1 * np.sin(theta)
        Z.append(zj)
    Z = torch.cat(Z, 0)
    # decode many times and select the most common one
    G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    names = []
    for j in range(0, interpolate_number + 1):
        namej = 'graph_interpolate_{}_of_{}'.format(j, interpolate_number)
        namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True,
                         data_type=args.data_type)
        names.append(namej)
    # draw figures with the same height
    new_name = os.path.join(args.res_dir, args.data_name +
                            '_{}_interpolate_exp3_ensemble_epoch{}.pdf'.format(args.model, epoch))
    combine_figs_horizontally(names, new_name)


def smoothness_exp(epoch, gap=0.05):
    print('Smoothness experiments around a latent vector')
    smoothness_res_dir = os.path.join(args.res_dir, 'smoothness')
    g0 = []
    if not os.path.exists(smoothness_res_dir):
        os.makedirs(smoothness_res_dir)

        # z0 = torch.zeros(1, model.nz).to(device)  # use all-zero vector as center

    if args.data_type == 'ENAS_FULL':
        g_str = '4 4 0 3 0 0 5 0 0 1 2 0 0 0 0 5 0 0 0 1 0'  # a 6-layer network
        row = [int(x) for x in g_str.split()]
        row = flat_ENAS_to_nested(row, model.max_n_vertex - 2)
        if args.model.startswith('SVAE'):
            g0, _ = decode_ENAS_to_tensor(row, n_types=model.max_n_vertex - 2)
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])
        elif args.model.startswith('DVAE'):
            # g0, _ = decode_ENAS_to_igraph(row)
            adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g = decode_ENAS_to_Full(row)
            g0.append((0, adj_mat, adj_mat_N, nodes, ops, inci_lb_T, inci_lb_H, inci_mat_T, edges, edges_t, weight_tensor, weight_tensor_T, weight_tensor_H, g))
    elif args.data_type == 'BN':
        g0 = train_data[20][0]
        if args.model.startswith('SVAE'):
            g0 = g0.to(device)
            g0 = model._collate_fn([g0])

    _, _, z0 = model.encode(g0[0])

    # select two orthogonal directions in latent space
    tmp = np.random.randn(z0.shape[1], z0.shape[1])
    Q, R = qr(tmp)
    dir1 = torch.FloatTensor(tmp[0:1, :]).to(device)
    dir2 = torch.FloatTensor(tmp[1:2, :]).to(device)

    # generate architectures along two orthogonal directions
    grid_size = 13
    grid_size = 9
    mid = grid_size // 2
    Z = []
    pbar = tqdm.tqdm(range(grid_size ** 2))
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        zij = z0 + dir1 * (i - mid) * gap + dir2 * (j - mid) * gap
        Z.append(zij)
    Z = torch.cat(Z, 0)
    if True:
        G, _ = decode_from_latent_space(Z, model, return_igraph=True, data_type=args.data_type)
    else:  # decode by 3 batches in case of GPU out of memory
        Z0, Z1, Z2 = Z[:len(Z) // 3, :], Z[len(Z) // 3:len(Z) // 3 * 2, :], Z[len(Z) // 3 * 2:, :]
        G = []
        G += decode_from_latent_space(Z0, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z1, model, return_igraph=True, data_type=args.data_type)[0]
        G += decode_from_latent_space(Z2, model, return_igraph=True, data_type=args.data_type)[0]
    names = []
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        pbar.set_description('Drawing row {}/{}, col {}/{}...'.format(i + 1,
                                                                      grid_size, j + 1, grid_size))
        nameij = 'graph_smoothness{}_{}'.format(i, j)
        nameij = plot_DAG(G[idx], smoothness_res_dir, nameij, data_type=args.data_type)
        names.append(nameij)
    # fig = plt.figure(figsize=(200, 200))
    if args.data_type == 'ENAS':
        fig = plt.figure(figsize=(50, 50))
    elif args.data_type == 'BN':
        fig = plt.figure(figsize=(30, 30))

    nrow, ncol = grid_size, grid_size
    for ij, nameij in enumerate(names):
        imgij = mpimg.imread(nameij)
        fig.add_subplot(nrow, ncol, ij + 1)
        plt.imshow(imgij)
        plt.axis('off')
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.savefig(os.path.join(args.res_dir,
                             args.data_name + '_{}_smoothness_ensemble_epoch{}_gap={}_small.pdf'.format(
                                 args.model, epoch, gap)), bbox_inches='tight')


def sample():
    print('Sampling starting ...')
    sampled = model.generate_sample(args.sample_number)
    for i, g in enumerate(sampled):
        namei = 'graph_sample{}'.format(i)
        plot_DAG(g, args.res_dir, namei, data_type=args.data_type)

    print('Done!')

'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')

if not args.link_prediction:
    if os.path.exists(loss_name) and not args.keep_old:
        os.remove(loss_name)

    if args.only_test:
        epoch = args.continue_from
        visualize_recon(300)
        pdb.set_trace()

    start_epoch = args.continue_from if args.continue_from is not None else 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.predictor:
            train_loss, ver_ls, adj_ls, inci_ls, edge_ls, kld_loss, pred_loss = train(epoch)
        else:
            train_loss, ver_ls, adj_ls, inci_ls, edge_ls, kld_loss = train(epoch)
            pred_loss = 0.0
        with open(loss_name, 'a') as loss_file:
            loss_file.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                train_loss / len(train_data),
                ver_ls / len(train_data),
                adj_ls / len(train_data),
                inci_ls / len(train_data),
                edge_ls / len(train_data),
                kld_loss / len(train_data),
            ))
        scheduler.step(train_loss)
        if epoch % args.save_interval == 0:
            print("save current model...")
            model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
            scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            print("visualize reconstruction examples...")
            visualize_recon(epoch)
            print("extract latent representations...")
            # save_latent_representations(epoch)
            print("sample from prior...")
            sampled = model.generate_sample(args.sample_number)
            for i, g in enumerate(sampled):
                namei = 'graph_{}_sample{}'.format(epoch, i)
                plot_DAG(g, args.res_dir, namei, data_type=args.data_type)
            print("plot train loss...")
            losses = np.loadtxt(loss_name)
            if losses.ndim == 1:
                continue
            fig = plt.figure()
            num_points = losses.shape[0]
            plt.plot(range(1, num_points + 1), losses[:, 0], label='Total')
            plt.plot(range(1, num_points + 1), losses[:, 1], label='Recon')
            plt.plot(range(1, num_points + 1), losses[:, 2], label='KLD')
            plt.plot(range(1, num_points + 1), losses[:, 3], label='Pred')
            plt.xlabel('Epoch')
            plt.ylabel('Train loss')
            plt.legend()
            plt.savefig(loss_plot_name)
            _, _, _ = test_during_train()

    sample()
    exit()

    '''Testing begins here'''
    if args.predictor:
        Nll, acc, pred_rmse = test()
    else:
        Nll, acc, _ = test()
        pred_rmse = 0
    r_valid, r_unique, r_novel = prior_validity(scale_to_train_range=True)
    with open(test_results_name, 'a') as result_file:
        result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) +
                          " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
                              r_unique, r_novel, pred_rmse))
    interpolation_exp2(epoch)
    smoothness_exp(epoch)
    interpolation_exp3(epoch)

    pdb.set_trace()

if args.link_prediction:
    link_pred()
    Nll, acc = test()
    pred_rmse = 0
    r_valid, r_unique, r_novel = prior_validity(scale_to_train_range=True)
    with open(test_results_name, 'a') as result_file:
        result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) +
                          " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
                              r_unique, r_novel, pred_rmse))
    interpolation_exp2(epoch)
    smoothness_exp(epoch)
    interpolation_exp3(epoch)
