import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from basic.SetAbstraction import *
from basic.constant import *
import numpy as np
from basic.common import minPointNet
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


def distance(pose):
    pose_dis = pose.clone().detach()
    pose_dis = torch.norm(pose_dis, dim=-1, p=2, keepdim=True)
    return pose_dis


class GATConv(nn.Module):
    def __init__(self, output_dim=64, num_heads=1, dropout=0.5, alpha=0.2, case1=False):
        super(GATConv, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.case = case1
        if self.case:
            self.attention = nn.ModuleList([nn.Conv2d(2 * output_dim, 1, kernel_size=1) for _ in range(num_heads)])
        else:
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            mlp = [output_dim, 1]
            last_channel = output_dim
            for out_channel_ in mlp:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel_, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel_))
                last_channel = out_channel_
                # self.attention = nn.ModuleList([nn.Conv2d(1 * output_dim, 1, kernel_size=1) for _ in range(num_heads)])
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, edge_index):
        # x should have the size of B*D*N
        # edge_index should have the size B*N*k
        if self.case:
            index = torch.arange(edge_index.shape[1], device=x.device).unsqueeze(dim=0).unsqueeze(dim=2)
            index = index.repeat(edge_index.shape[0], 1, 1)
            # add the self-loop
            edge_index_self_loop = torch.cat([index, edge_index], dim=2)
            edge_index_self_loop = edge_index.clone().detach()
            index = index.repeat(1, 1, edge_index_self_loop.shape[2])
            x = x.permute(0, 2, 1)  # has B*N*D
            features_expand = x.unsqueeze(dim=3)
            features_expand = features_expand.repeat(1, 1, 1, edge_index_self_loop.shape[2])  # B*N*D*k
            features_nn = torch.cat(
                [x[i, edge_index_self_loop[i, :, :], :] for i in torch.arange(edge_index_self_loop.shape[0])], dim=0)
            features_nn = features_nn.reshape(edge_index_self_loop.shape[0], edge_index_self_loop.shape[1],
                                              edge_index_self_loop.shape[2], -1)  # B*N*K*D
            features_nn = features_nn.permute(0, 1, 3, 2)  # B*N*D*k
            cat_features = torch.cat([features_expand, features_nn], dim=2).permute(0, 2, 1, 3)
            att = F.leaky_relu(self.attention[0](cat_features), negative_slope=self.alpha)
            alpha = self.softmax(att)
            alpha = F.dropout(alpha, p=self.dropout)
            alpha = alpha.permute(0, 2, 3, 1)  # B*N*K*1
            features_nn = features_nn.permute(0, 1, 3, 2)  # B,N,K,64
            h = torch.sum(features_nn * alpha, dim=2)  # B*N*K*D -> B*N*D
        else:
            x3 = x.clone().detach()
            for i, conv2d in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                x3 = F.leaky_relu(bn(conv2d(x3)), negative_slope=self.alpha)
            att = x3
            # cat_features = x
            # att = F.leaky_relu(self.attention[0](cat_features), negative_slope=self.alpha)
            alpha = self.softmax(att)
            # alpha = F.dropout(alpha, p=self.dropout)
            alpha = alpha.permute(0, 2, 3, 1)  # B*N*K*1
            features_nn = x.permute(0, 2, 3, 1)  # B,N,K,64
            h = torch.sum(features_nn * alpha, dim=2)  # B*N*K*D -> B*N*D

        return h


class featureTransformation(nn.Module):

    def __init__(self, in_channel, mlp, out_channel=64, use_gat=False):
        super(featureTransformation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.use_GAT = use_gat
        last_channel = in_channel
        self.case1 = False
        self.softmax = nn.Softmax(dim=3)
        if self.use_GAT:
            self.GATConv = GATConv(case1=self.case1)
        for out_channel_ in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel_, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel_))
            last_channel = out_channel_

    def forward(self, x, y, edge_index=None):
        B, N, D = x.shape
        x1 = torch.split(x, FPH_DIM, dim=2)
        x1 = torch.stack(x1, dim=3)  # B,N,4, k
        y1 = y.unsqueeze(dim=3)
        y1 = y1.repeat(1, 1, 1, x1.shape[3])
        x3 = torch.cat([x1, y1], dim=2).permute(0, 2, 1, 3)  # B,N,1+4,K --> B,5,N,K
        for i, conv2d in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x3 = F.relu(bn(conv2d(x3)))
        if self.use_GAT:
            if self.case1:
                x3 = x3.max(dim=-1, keepdim=False)[0]  # B,64,N
            x3 = self.GATConv(x3, edge_index)
        else:
            x3 = x3.max(dim=-1, keepdim=False)[0]  # B,64,N
            x3 = x3.transpose(2, 1)
        return x3


class featureTransformation3D(nn.Module):

    def __init__(self, maxdegree, mlp, in_channel, out_channel=64):
        super(featureTransformation3D, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel_ in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel_, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel_))
            last_channel = out_channel_

    def forward(self, x, y):
        x1 = torch.split(x, FPH_DIM, dim=3)  # B,N,K,D(4*K-1) --> B,N,K,4,K-1
        x1 = torch.stack(x1, dim=4)  # --> B,N,K,4,K-1
        if y is not None:
            y1 = y.unsqueeze(dim=4)
            y1 = y1.repeat(1, 1, 1, 1, x1.shape[4])
            x3 = torch.cat([x1, y1], dim=3).permute(0, 3, 2, 1, 4)  # B,N,K,1+4,K-1 --> B,1+4,K,N,K-1
        else:
            x3 = x1.permute(0, 3, 2, 1, 4)  # B,N,K,4,K-1 --> B,4,K,N,K-1

        for i, conv3d in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x3 = F.relu(bn(conv3d(x3)))
        x3 = x3.max(dim=-1, keepdim=False)[0]  # B,64,K,N
        x3 = x3.transpose(3, 1)  # B,N,K,64
        return x3


class featureTransformNet(nn.Module):
    def __init__(self, maxdegree):
        super(featureTransformNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1 = nn.Sequential(nn.Conv1d(fpfhDim + maxdegree + 1, 64, kernel_size=1), self.bn1,
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1), self.bn1,
                                   nn.ReLU())

    def forward(self, x, y):
        x3 = torch.cat([x, y], dim=2).permute(0, 2, 1)  # B,N,1+4,K --> B,5,N,K
        x3 = self.conv1(x3)
        x3 = x3.transpose(2, 1)
        return x3
