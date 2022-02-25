import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
from scipy.spatial.transform import Rotation as R
import torch_geometric.nn
from torch.nn import Sequential as Seq, ReLU, BatchNorm1d as BN
import shutil
from basic.constant import *


def unitSpher(points):
    norm = torch.norm(points, dim=2, p=2, keepdim=True)
    points_ = points / norm
    return points_


def scalingPoints(points):
    rlist = torch.norm(points, dim=2, p=2, keepdim=True)
    maxrlist = torch.max(rlist, dim=1, keepdim=True)[0]
    maxrlist = maxrlist.repeat(1, points.shape[1], 1)
    points_ = points / maxrlist
    return (points_)

def singleScalingPoint(point):
    rlist = torch.norm(point, dim=1, p=2, keepdim=True)
    maxrlist = torch.min(rlist, dim=0, keepdim=True)[0]
    maxrlist = maxrlist.repeat(1, point.shape[1])
    points_ = point / maxrlist
    return points_

def zeroCenter(points):# for B*N*3
    mean = points.mean(1).reshape(points.shape[0], 1, points.shape[2])
    points_ = points - mean
    return points_

def singleZeroCenter(points):# for B*N*3
    mean = points.mean(0).reshape(1, points.shape[1])
    points_ = points - mean
    return points_


def minPointNet(channels, batch_norm=True):
    return Seq(*[Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False), ReLU(), BN(channels[i])) for i in
                 range(1, len(channels))])


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # creat a one-hot vector
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def generateRandomMatrix(batch_size):
    mat = []
    for i in range(batch_size):
        deg = np.random.rand(3, 1) * 180
        rx = R.from_euler('x', deg[0], degrees=True)
        ry = R.from_euler('y', deg[1], degrees=True)
        rz = R.from_euler('z', deg[2], degrees=True)
        r = np.matmul(np.matmul(rz.as_dcm(), ry.as_dcm()), rx.as_dcm())
        mat.append(r)
    mat = np.concatenate(mat, axis=0)
    # print(mat)
    mat = mat.reshape(batch_size, 3, 3)
    # print("after ")
    # print(mat)
    return mat


def rotate_pointcloud(batch_size, x):
    rotmat = generateRandomMatrix(batch_size)
    maxPoints = x.shape[1]

    newMat = np.zeros([batch_size, maxPoints, 3])
    for i in range(batch_size):
        temp = rotmat[i, :, :].reshape(3, 3)
        points = x[i, :, :].reshape(-1, 3)
        points = points.transpose(1, 0)
        newMat[i, :, :] = (np.dot(temp, points)).transpose(1, 0)

    return newMat


class TorchClusteringKNN:
    def __init__(self, points, K, includeSelf=False):
        self.K = K
        self.edge_index_knn = torch_geometric.nn.knn_graph(points, k=K, loop=includeSelf)

    def query(self, index):
        return self.edge_index_knn[:, self.edge_index_knn[1, :] == index]


class BatchKnn:  # B,N,D
    def __init__(self, points, maxKnn, includeSelf=False, queryPoints=None):
        if ~includeSelf:
            self.maxKnn = maxKnn + 1
        else:
            self.maxKnn = maxKnn
        self.points = points
        self.pointCRange = self.points.reshape(-1, self.points.shape[2])
        batchIndex = torch.arange(self.points.shape[0])
        batch = batchIndex.repeat(self.points.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
        #batch = batchIndex.repeat(self.points.shape[1], 1).transpose(0, 1).reshape(-1).cuda()
        if queryPoints is not None:
            self.queryPoints = queryPoints
            batch_query = batchIndex.repeat(queryPoints.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
            #batch_query = batchIndex.repeat(queryPoints.shape[1], 1).transpose(0, 1).reshape(-1).cuda()
            queryPointsCRange = self.queryPoints.reshape(-1, queryPoints.shape[2])
            assign_index = torch_geometric.nn.knn(self.pointCRange, queryPointsCRange, self.maxKnn, batch, batch_query)
        else:
            assign_index = torch_geometric.nn.knn(self.pointCRange, self.pointCRange, self.maxKnn, batch, batch)
        localRegion = self.pointCRange[assign_index[1, :], :]
        # B*N'*k*dim
        if queryPoints is not None:
            self.cluster = localRegion.reshape(self.queryPoints.shape[0], self.queryPoints.shape[1], self.maxKnn,
                                               self.queryPoints.shape[2])
            self.indices = assign_index[1, :].reshape(self.queryPoints.shape[0], self.queryPoints.shape[1], self.maxKnn)
        else:
            self.cluster = localRegion.reshape(self.points.shape[0], self.points.shape[1], self.maxKnn,
                                               self.points.shape[2])
            self.indices = assign_index[1, :].reshape(self.points.shape[0], self.points.shape[1], self.maxKnn)
        self.indices = self.indices % self.points.shape[1]

        if ~includeSelf:
            self.remove_self_loop()

    def remove_self_loop(self):
        self.cluster = self.cluster[:, :, 1:, :]
        self.indices = self.indices[:, :, 1:]

    def query(self, index):
        return self.cluster[:, index, :, :]

    def queryIndics(self, index):
        return self.query(index), self.indices[:, index, :]


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    print(farthest)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # print("distance:", distance.shape)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        # print(mask.shape)
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        print(farthest)
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


if __name__ == "__main__":
    x = torch.rand(8, 20, 3)
    y = torch.rand(8, 4, 3)
    knnT = BatchKnn(x, 5, queryPoints=y)
    # clu, indices = knnT.queryIndics([0, 1])
    # print(clu[0, :, :, :])
    # x1 = torch.stack([x[i, indices[i, :], :] for i in range(2)])
    # print(x1[0, :, :, :])

    # x1=x[indices]
