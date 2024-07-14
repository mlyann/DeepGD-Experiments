## custom
from utils import utils, vis
# from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import criteria as C
import quality as Q
# import gd2
from gd2 import GD2
import utils.weight_schedule as ws

## third party
import networkx as nx
# from PIL import Image
from natsort import natsorted

### numeric
import numpy as np
# import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F

### vis
import tqdm
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.colors import LinearSegmentedColormap
# from mpl_toolkits import mplot3d
# from matplotlib import collections  as mc
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')


## sys
from collections import defaultdict
import random
import time
from glob import glob
import math
import os
from pathlib import Path
import itertools
import pickle as pkl



import torch
import os

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

device = "cpu"
for backend, device_name in {
    torch.backends.mps: "mps",
    torch.cuda: "cuda",
}.items():
    if backend.is_available():
        device = device_name

import random

import torch
import torch_geometric as pyg
from tqdm.auto import *

DATA_ROOT = "data"

import os
import re

import torch
import torch_geometric as pyg
import networkx as nx
import numpy as np


class RomeDataset(pyg.data.InMemoryDataset):
    def __init__(self, *,
                 url='http://www.graphdrawing.org/download/rome-graphml.tgz',
                 root=f'{DATA_ROOT}/Rome',
                 layout_initializer=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.url = url
        self.initializer = layout_initializer or nx.drawing.random_layout
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        metafile = "rome/Graph.log"
        if os.path.exists(metadata_path := f'{self.raw_dir}/{metafile}'):
            return list(map(lambda f: f'rome/{f}.graphml',
                            self.get_graph_names(metadata_path)))
        else:
            return [metafile]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @classmethod
    def get_graph_names(cls, logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def process_raw(self):
        graphmls = sorted(self.raw_paths,
                          key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            if nx.is_connected(G):
                yield nx.convert_node_labels_to_integers(G)

    def convert(self, G):
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        init_pos = torch.tensor(np.array(list(self.initializer(G).values())))
        full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
        raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
        full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
            torch.tensor(full_edges).T, torch.tensor(attr_d)
        ))
        k = 1 / d ** 2
        full_edge_attr = torch.stack([d, k], dim=-1)
        return pyg.data.Data(
            G=G,
            x=init_pos,
            init_pos=init_pos,
            edge_index=full_edge_index,
            edge_attr=full_edge_attr,
            raw_edge_index=raw_edge_index,
            full_edge_index=full_edge_index,
            full_edge_attr=full_edge_attr,
            d=d,
            n=G.number_of_nodes(),
            m=G.number_of_edges(),
        )

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        data_list = map(self.convert, self.process_raw())

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])

#dataset = RomeDataset()
dataset = RomeDataset(layout_initializer=nx.spectral_layout)

from itertools import chain

import torch
from torch import nn
import torch_geometric as pyg

def l2_normalize(x, return_norm=False, eps=1e-5):
    if type(x) is torch.Tensor:
        norm = x.norm(dim=1).unsqueeze(dim=1)
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit_vec = x / (norm + eps)
    if return_norm:
        return unit_vec, norm
    else:
        return unit_vec


def get_edges(node_pos, batch):
    edges = node_pos[batch.edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_full_edges(node_pos, batch):
    edges = node_pos[batch.full_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_raw_edges(node_pos, batch):
    edges = node_pos[batch.raw_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_per_graph_property(batch, property_getter):
    return torch.tensor(list(map(property_getter, batch.to_data_list())),
                        device=batch.x.device)


def map_node_indices_to_graph_property(batch, node_index, property_getter):
    return get_per_graph_property(batch, property_getter)[batch.batch][node_index]


def map_node_indices_to_node_degrees(real_edges, node_indices):
    node, degrees = np.unique(real_edges[:, 0].detach().cpu().numpy(), return_counts=True)
    return torch.tensor(degrees[node_indices], device=real_edges.device)


def get_counter_clockwise_sorted_angle_vertices(edges, pos):
    if type(pos) is torch.Tensor:
        edges = edges.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
    u, v = edges[:, 0], edges[:, 1]
    diff = pos[v] - pos[u]
    diff_normalized = l2_normalize(diff)
    # get cosine angle between uv and y-axis
    cos = diff_normalized @ np.array([[1],[0]])
    # get radian between uv and y-axis
    radian = np.arccos(cos) * np.expand_dims(np.sign(diff[:, 1]), axis=1)
    # for each u, sort edges based on the position of v
    sorted_idx = sorted(np.arange(len(edges)), key=lambda e: (u[e], radian[e]))
    sorted_v = v[sorted_idx]
    # get start index for each u
    idx = np.unique(u, return_index=True)[1]
    roll_idx = np.arange(1, len(u) + 1)
    roll_idx[np.roll(idx - 1, -1)] = idx
    rolled_v = sorted_v[roll_idx]
    return np.stack([u, sorted_v, rolled_v]).T[sorted_v != rolled_v]


def get_radians(pos, batch,
                return_node_degrees=False,
                return_node_indices=False,
                return_num_nodes=False,
                return_num_real_edges=False):
    real_edges = batch.raw_edge_index.T
    angles = get_counter_clockwise_sorted_angle_vertices(real_edges, pos)
    u, v1, v2 = angles[:, 0], angles[:, 1], angles[:, 2]
    e1 = l2_normalize(pos[v1] - pos[u])
    e2 = l2_normalize(pos[v2] - pos[u])
    radians = (e1 * e2).sum(dim=1).acos()
    result = (radians,)
    if return_node_degrees:
        degrees = map_node_indices_to_node_degrees(real_edges, u)
        result += (degrees,)
    if return_node_indices:
        result += (u,)
    if return_num_nodes:
        node_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: g.num_nodes)
        result += (node_counts,)
    if return_num_real_edges:
        edge_counts = map_node_indices_to_graph_property(batch, angles[:,0], lambda g: len(g.raw_edge_index.T))
        result += (edge_counts,)
    return result[0] if len(result) == 1 else result

def generate_rand_pos(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2).mul(2).sub(1)

class GNNLayer(nn.Module):
    def __init__(self,
                 nfeat_dims,
                 efeat_dim,
                 aggr,
                 edge_net=None,
                 dense=False,
                 bn=True,
                 act=True,
                 dp=None,
                 root_weight=True,
                 skip=True):
        super().__init__()
        try:
            in_dim = nfeat_dims[0]
            out_dim = nfeat_dims[1]
        except:
            in_dim = nfeat_dims
            out_dim = nfeat_dims
        self.enet = nn.Linear(efeat_dim, in_dim * out_dim) if edge_net is None and efeat_dim > 0 else edge_net
        self.conv = pyg.nn.NNConv(in_dim, out_dim, nn=self.enet, aggr=aggr, root_weight=root_weight)
        self.dense = nn.Linear(out_dim, out_dim) if dense else nn.Identity()
        self.bn = pyg.nn.BatchNorm(out_dim) if bn else nn.Identity()
        self.act = nn.LeakyReLU() if act else nn.Identity()
        self.dp = dp and nn.Dropout(dp) or nn.Identity()
        self.skip = skip
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, v, e, data):
        v_ = v
        v = self.conv(v, data.edge_index, e)
        v = self.dense(v)
        v = self.bn(v)
        v = self.act(v)
        v = self.dp(v)
        return v + self.proj(v_) if self.skip else v

class GNNBlock(nn.Module):
    def __init__(self,
                 feat_dims,
                 efeat_hid_dims=[],
                 efeat_hid_act=nn.LeakyReLU,
                 efeat_out_act=nn.Tanh,
                 bn=False,
                 act=True,
                 dp=None,
                 aggr='mean',
                 root_weight=True,
                 static_efeats=2,
                 dynamic_efeats='skip',
                 euclidian=False,
                 direction=False,
                 n_weights=0,
                 residual=False):
        '''
        dynamic_efeats: {
            skip: block input to each layer,
            first: block input to first layer,
            prev: previous layer output to next layer,
            orig: original node feature to each layer
        }
        '''
        super().__init__()
        self.static_efeats = static_efeats
        self.dynamic_efeats = dynamic_efeats
        self.euclidian = euclidian
        self.direction = direction
        self.n_weights = n_weights
        self.residual = residual
        self.gnn = nn.ModuleList()
        self.n_layers = len(feat_dims) - 1

        for idx, (in_feat, out_feat) in enumerate(zip(feat_dims[:-1], feat_dims[1:])):
            direction_dim = (feat_dims[idx] if self.dynamic_efeats == 'prev'
                             else 2 if self.dynamic_efeats == 'orig'
                             else feat_dims[0])
            in_efeat_dim = self.static_efeats
            if self.dynamic_efeats != 'first':
                in_efeat_dim += self.euclidian + self.direction * direction_dim + self.n_weights
            edge_net = nn.Sequential(*chain.from_iterable(
                [nn.Linear(idim, odim),
                 nn.BatchNorm1d(odim),
                 act()]
                for idim, odim, act in zip([in_efeat_dim] + efeat_hid_dims,
                                           efeat_hid_dims + [in_feat * out_feat],
                                           [efeat_hid_act] * len(efeat_hid_dims) + [efeat_out_act])
            ))
            self.gnn.append(GNNLayer(nfeat_dims=(in_feat, out_feat),
                                     efeat_dim=in_efeat_dim,
                                     edge_net=edge_net,
                                     bn=bn,
                                     act=act,
                                     dp=dp,
                                     aggr=aggr,
                                     root_weight=root_weight,
                                     skip=False))

    def _get_edge_feat(self, pos, data, euclidian=False, direction=False, weights=None):
        e = data.edge_attr[:, :self.static_efeats]
        if euclidian or direction:
            start_pos, end_pos = get_edges(pos, data)
            v, u = l2_normalize(end_pos - start_pos, return_norm=True)
            if euclidian:
                e = torch.cat([e, u], dim=1)
            if direction:
                e = torch.cat([e, v], dim=1)
        if weights is not None:
            w = weights.repeat(len(e), 1)
            e = torch.cat([e, w], dim=1)
        return e

    def forward(self, v, data, weights=None):
        vres = v
        for layer in range(self.n_layers):
            vsrc = (v if self.dynamic_efeats == 'prev'
                    else data.pos if self.dynamic_efeats == 'orig'
                    else vres)
            get_extra = not (self.dynamic_efeats == 'first' and layer != 0)
            e = self._get_edge_feat(vsrc, data,
                                    euclidian=self.euclidian and get_extra,
                                    direction=self.direction and get_extra,
                                    weights=weights if get_extra and self.n_weights > 0 else None)
            v = self.gnn[layer](v, e, data)
        return v + vres if self.residual else v

class DeepGD(nn.Module):
    def __init__(self,
                 num_blocks=9,
                 num_layers=3,
                 num_enet_layers=2,
                 layer_dims=None,
                 n_weights=0,
                 dynamic_efeats='skip',
                 euclidian=True,
                 direction=True,
                 residual=True,
                 normalize=None):
        super().__init__()

        self.in_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[2, 8, 8 if layer_dims is None else layer_dims[0]], bn=True, dp=0.2, static_efeats=2)
        ])
        self.hid_blocks = nn.ModuleList([
            GNNBlock(feat_dims=layer_dims or ([8] + [8] * num_layers),
                     efeat_hid_dims=[16] * (num_enet_layers - 1),
                     bn=True,
                     act=True,
                     dp=0.2,
                     static_efeats=2,
                     dynamic_efeats=dynamic_efeats,
                     euclidian=euclidian,
                     direction=direction,
                     n_weights=n_weights,
                     residual=residual)
            for _ in range(num_blocks)
        ])
        self.out_blocks = nn.ModuleList([
            GNNBlock(feat_dims=[8 if layer_dims is None else layer_dims[-1], 8], bn=True, static_efeats=2),
            GNNBlock(feat_dims=[8, 2], act=False, static_efeats=2)
        ])
        self.normalize = normalize

    def forward(self, data, weights=None, output_hidden=False, numpy=False):
        v = data.init_pos if data.init_pos is not None else generate_rand_pos(len(data.x)).to(data.x.device)
        if self.normalize is not None:
            v = self.normalize(v, data)

        hidden = []
        for block in chain(self.in_blocks,
                           self.hid_blocks,
                           self.out_blocks):
            v = block(v, data, weights)
            if output_hidden:
                hidden.append(v.detach().cpu().numpy() if numpy else v)
        if not output_hidden:
            vout = v.detach().cpu().numpy() if numpy else v
            if self.normalize is not None:
                vout = self.normalize(vout, data)

        return hidden if output_hidden else vout

model = DeepGD().to(device)

PATH = "/work/mlyang721/deepg/model_scripted_32400.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

import torch_scatter

class EdgeVar(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        edge_idx = batch.raw_edge_index.T
        start, end = get_raw_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_var = eu.sub(1).square()
        index = batch.batch[batch.raw_edge_index[0]]
        graph_var = torch_scatter.scatter(edge_var, index, reduce="mean")
        return graph_var if self.reduce is None else self.reduce(graph_var)

class IncidentAngle(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        theta, degrees, indices = get_radians(node_pos, batch,
                                              return_node_degrees=True,
                                              return_node_indices=True)
        phi = degrees.float().pow(-1).mul(2*np.pi)
        angle_l1 = phi.sub(theta).abs()
        index = batch.batch[indices]
        graph_l1 = torch_scatter.scatter(angle_l1, index)
        return graph_l1 if self.reduce is None else self.reduce(graph_l1)

class Occlusion(nn.Module):
    def __init__(self, gamma=1, reduce=torch.mean):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        edge_occusion = eu.mul(-self.gamma).exp()
        index = batch.batch[batch.edge_index[0]]
        graph_occusion = torch_scatter.scatter(edge_occusion, index)
        return graph_occusion if self.reduce is None else self.reduce(graph_occusion)

class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        # print(node_pos)
        # print(batch)
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)

class TSNEScore(nn.Module):
    def __init__(self, sigma=1, reduce=torch.mean):
        super().__init__()
        self.sigma = sigma
        self.reduce = reduce

    def forward(self, node_pos, batch):
        p = batch.full_edge_attr[:, 0].div(-2 * self.sigma**2).exp()
        sum_src = torch_scatter.scatter(p, batch.full_edge_index[0])[batch.full_edge_index[0]]
        sum_dst = torch_scatter.scatter(p, batch.full_edge_index[1])[batch.full_edge_index[1]]
        p = (p / sum_src + p / sum_dst) / (2 * batch.n[batch.batch[batch.edge_index[0]]])
        start, end = get_full_edges(node_pos, batch)
        eu = end.sub(start).norm(dim=1)
        index = batch.batch[batch.full_edge_index[0]]
        q = 1 / (1 + eu.square())
        q /= torch_scatter.scatter(q, index)[index]
        edge_kl = (p.log() - q.log()).mul(p)
        graph_kl = torch_scatter.scatter(edge_kl, index)
        return graph_kl if self.reduce is None else self.reduce(graph_kl)

import torch
import torch.nn as nn
import numpy as np

class GabrielGraphLoss(nn.Module):
    def __init__(self, margin=0.01, reduce=torch.mean):
        super().__init__()
        self.margin = margin
        self.reduce = reduce

    def forward(self, node_pos, batch):
        n = node_pos.size(0)
        full_edge_index = torch.combinations(torch.arange(n), r=2).t().to(node_pos.device)
        edge_pos = node_pos[full_edge_index]
        midpoints = edge_pos.mean(dim=1)
        radii = (edge_pos[:, 0, :] - edge_pos[:, 1, :]).norm(dim=1, p=2) / 2 + self.margin
        distances = torch.cdist(node_pos, midpoints)
        energy = torch.relu(radii.unsqueeze(0) - distances).pow(2)
        if self.reduce == torch.mean:
            return energy.mean()
        elif self.reduce == torch.sum:
            return energy.sum()
        else:
            return energy

# Example of how to instantiate and use this class
# node_pos would be a tensor of node positions, batch is the graph batch
# model = GabrielGraphLoss(margin=0.01, reduce=torch.mean)
# loss = model(node_positions, batch)

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossingNumberLoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super(CrossingNumberLoss, self).__init__()
#         # 初始化参数w和b
#         self.w = nn.Parameter(torch.randn(2, 1))
#         self.b = nn.Parameter(torch.randn(1))
#         self.gamma = gamma

#     def forward(self, node_pos, batch):
#         total_loss = 0.0
#         d_max = torch.max(batch.batch).item()+1
#         n_edges = batch.full_edge_index.size(1)
#         for i in range(n_edges):
#             for j in range(i + 1, n_edges):
#                 e1_idx = batch.full_edge_index[:, i]
#                 e2_idx = batch.full_edge_index[:, j]
#                 # not same edges
#                 if len(torch.unique(torch.cat((e1_idx, e2_idx)))) < 4:
#                     continue
#                 pos_e1 = node_pos[e1_idx]
#                 pos_e2 = node_pos[e2_idx]
#                 x1, x2 = pos_e1[0] - pos_e1[1], pos_e2[0] - pos_e2[1]
#                 x = torch.stack([x1, x2], dim=0)
#                 logits = x @ self.w + self.b

#                 target = torch.tensor([1, -1], dtype=torch.float32, device=node_pos.device)
#                 margin_loss = F.relu(1 - target * logits).mean()

#                 total_loss += margin_loss

#         return total_loss / (n_edges * (n_edges - 1) / 2)

import torch
import torch.nn as nn

class CrossingNumberLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super(CrossingNumberLoss, self).__init__()
        self.threshold = threshold

    def forward(self, node_pos, batch):
        edge_index = batch.edge_index
        n_edges = edge_index.size(1)
        total_loss = 0.0

        edge_vectors = node_pos[edge_index[1]] - node_pos[edge_index[0]]
        norms = torch.norm(edge_vectors, dim=1, keepdim=True)
        edge_vectors = edge_vectors / norms.clamp(min=1e-6)

        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                cos_angle = torch.dot(edge_vectors[i], edge_vectors[j])
                if torch.abs(cos_angle) > self.threshold:
                    total_loss += 1

        return total_loss / (n_edges * (n_edges - 1) / 2)

criteria = {
    Stress(): 1,
    EdgeVar(): 0,
    Occlusion(): 0,
    IncidentAngle(): 0,
    TSNEScore(): 0,
    GabrielGraphLoss():0,
    CrossingNumberLoss():0
    }
optim = torch.optim.AdamW(model.parameters())

datalist = list(dataset)
random.seed(12345)
random.shuffle(datalist)

batch_size = 32

# train_loader = pyg.loader.DataLoader(datalist[:10000], batch_size=batch_size, shuffle=True)
# val_loader = pyg.loader.DataLoader(datalist[11000:], batch_size=batch_size, shuffle=False)
test_loader = pyg.loader.DataLoader(datalist[10000:11000], batch_size=batch_size, shuffle=False)
train_loader = pyg.loader.DataLoader(datalist[:100], batch_size=batch_size, shuffle=True)
val_loader = pyg.loader.DataLoader(datalist[110:120], batch_size=batch_size, shuffle=False)
# test_loader = pyg.loader.DataLoader(datalist[100:110], batch_size=batch_size, shuffle=False)

# for epoch in range(1):
# for epoch in range(10):
for epoch in range(0):
    model.train()
    losses = []
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        model.zero_grad()
        loss = 0
        for c, w in criteria.items():
            loss += w * c(model(batch), batch)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    print(f'[Epoch {epoch}] Train Loss: {np.mean(losses)}')
    with torch.no_grad():
        model.eval()
        losses = []
        for batch in tqdm(val_loader, disable=True):
            batch = batch.to(device)
            #loss = criterion(model(batch), batch)
            loss = 0
            for c, w in criteria.items():
              loss += w * c(model(batch), batch)
            losses.append(loss.item())
        print(f'[Epoch {epoch}] Val Loss: {np.mean(losses)}')

# PATH = 'model_scripted.pt'
# torch.save(model.state_dict(), PATH)

def finda(X, sp_len):
    sum1 = torch.tensor(0.0)
    sum2 = torch.tensor(0.0)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i < j:
                x1T = torch.t(X[i])
                x2T = torch.t(X[j])
                eu = torch.nn.functional.pairwise_distance(x1T, x2T)
                if i in sp_len and j in sp_len[i]:
                    target_distance = sp_len[i][j]
                    sum1 += eu / target_distance
                    sum2 += (eu**2) / (target_distance**2)
    if sum2 != 0:  # Avoid division by zero
        a = sum1 / sum2
    else:
        a = torch.tensor(0.0)  # Default case, can adjust as necessary
    print(a)
    return a

def stress(X,sp_len):
    sum = torch.tensor([0.0])
    a = finda(X,sp_len)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i < j:
                x1T = torch.t(X[i])
                x2T = torch.t(X[j])
                eu = torch.mul(torch.nn.functional.pairwise_distance(x1T, x2T), torch.tensor(a, dtype=torch.float32))
                if i in sp_len and j in sp_len[i]:
                    # target_distance = sp_len[i][j]
                    # sum1 = sum1.add(eu.div(target_distance))
                    # sum2 = sum2.add(eu.square().div(target_distance**2))
                    target_distance = sp_len[i][j]
                    sum = sum.add(eu.sub(target_distance).square().div(target_distance**2))
    return sum


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph(G, rei, pos=None):
    g2 = nx.Graph()
    for n in G.nodes():
      g2.add_node(n)
    #print(rei.shape)
    edge_list = []
    for i in range(rei.shape[1]):
      edge_list.append((rei[0][i], rei[1][i]))
    #print(edge_list)
    for u in G.nodes():
      for v in G.nodes():
        if u<v:
          if (u,v) in edge_list:
            g2.add_edge(u,v)
    if pos==None:
      nx.draw_networkx(g2, pos=nx.spring_layout(g2, seed=42), with_labels=False, cmap="Set2")
      #print("spring positions:", nx.spring_layout(G, seed=42))
    else:
      nx.draw_networkx(g2, pos=pos, with_labels=False, cmap="Set2")
    plt.show()
    return g2

def optimize_graph_layout(G, graph_name='graph', max_iter=int(1e4)):
    criteria_weights = dict(
        stress=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]),
    )
    criteria = list(criteria_weights.keys())

    sample_sizes = dict(
        stress=16,
    )
    sample_sizes = {c: sample_sizes[c] for c in criteria}

    gd = GD2(G)
    result = gd.optimize(
        criteria_weights=criteria_weights,
        sample_sizes=sample_sizes,
        evaluate=set(criteria),
        max_iter=max_iter,
        time_limit=3600,  
        evaluate_interval=max_iter,
        evaluate_interval_unit='iter',
        vis_interval=-1,
        vis_interval_unit='sec',
        clear_output=True,
        grad_clamp=20,
        criteria_kwargs=dict(
            aspect_ratio=dict(target=[1, 1]),
        ),
        optimizer_kwargs=dict(mode='SGD', lr=2),
        scheduler_kwargs=dict(verbose=True),
    )

    # Extract the optimized positions
    pos = gd.pos.detach().numpy().tolist()
    pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}
    return pos_G

import time
#G = to_networkx(data, to_undirected=True)
#visualize_graph(G, color=data.y)
SGD = []
DeepGD = []
with torch.no_grad():
  model.eval()
  for batch in tqdm(test_loader, disable=True):
    batch = batch.to(device)
    #print(type(batch), batch)
    rei = batch.raw_edge_index.cpu().numpy()
    G = to_networkx(batch, to_undirected=True)
    CCs = list(nx.connected_components(G))
    print("Number of components:", len(CCs))
    start = time.time()
    out = model(batch)
    end = time.time()
    print(end-start)
    pos = dict()
    for i, x in enumerate(out):
      pos[i] = x.cpu().detach().numpy()
    for cc in CCs:
      cur_G = nx.induced_subgraph(G, cc)
    #   #print("Nodes:", cur_G.nodes())
    #   #print("pos:", pos)
    #   posSGD = optimize_graph_layout(cur_G)
    #   sp_lenSGD = dict(nx.all_pairs_shortest_path_length(visualize_graph(cur_G, rei, posSGD)))
    #   points = [value for key, value in posSGD.items()]
    #   SGD_model = torch.tensor(points, dtype=torch.float32)
    #   SGD.append(stress(SGD_model,sp_len))
      print("Model output:")
      sp_len = dict(nx.all_pairs_shortest_path_length(visualize_graph(cur_G, rei, pos)))
      points = [value for key, value in pos.items()]
      X_model = torch.tensor(points, dtype=torch.float32)
      DeepGD.append(stress(X_model,sp_len))
      print("Loss Of Stress:" + str(stress(X_model,sp_len)))
    #print(type(out), out.shape, out)
print(DeepGD)

