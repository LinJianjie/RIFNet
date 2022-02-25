import sys
import os

"""
import EstimateNormals should be done firstly, since the 
PyTorch is using -D_GLIBCXX_USE_CXX11_ABI=0
open3d is using  -D_GLIBCXX_USE_CXX11_ABI=1
"""
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# import basic.EstimateNormals as EN
from basic.importData import ModelNet40SphericalHarmonics
from torch.utils.data import DataLoader
from basic.common import TorchClusteringKNN, BatchKnn
import numpy as np
import torch
import time
import copy


# class PointFeatureMap:
#     # deprecated
#     def __init__(self, points, maxKnn, includeSelf=False):  # points=[N, dim]
#         self.includeSelf = includeSelf
#         self.maxKnn = maxKnn
#         self.points = points
#         self.estimateNormal = EN.EstimateNormals(points, maxKnn, includeSelf)
#
#     def computeSPF(self, queryPointIndex):
#         ps = self.points[queryPointIndex, 0, :]
#         ns = self.estimateNormal.normal[queryPointIndex, 1, :]
#         ind = self.estimateNormal.kdTree.query(queryPointIndex)[0, :]
#         vertices = self.points[ind, :]
#         simplifiedPointFeature = []
#         for i in range(vertices.shape[0]):
#             pt = vertices[i, :]
#             nt = self.estimateNormal.normal[ind[i], 1, :]
#             simplifiedPointFeature.append(computePairFeatures(ps, pt, ns, nt))
#         return torch.stack(simplifiedPointFeature)
#
#     def computePFH(self):
#         ind = torch.arange(0, self.points.shape[0])
#         x1 = ind.repeat(ind.shape[1], 1).transpose(0, 1).reshape(1, -1)
#         y1 = ind.repeat(ind.shape[1], 1).reshape(1, -1)
#         xy = torch.cat((x1, y1), 0)
#         tt = range(0, xy.shape[1], ind.shape[1] + 1)
#         tt2 = range(0, xy.shape[1])
#         tt3 = np.delete(np.asarray(tt2), np.asarray(tt))
#         verticesIndex = xy[:, tt3]
#         pair_vertices = self.points[verticesIndex, :]
#         j = 1
#         pfhFeature = []
#         pfh_local_list = []
#         # for i in range(pair_vertices.shape[0]):
#         for i, pair_vertex in enumerate(pair_vertices):
#             ps = pair_vertex[0, :]
#             pt = pair_vertex[1, :]
#             ns = self.estimateNormal.normal[verticesIndex[i, 0], 1, :]
#             nt = self.estimateNormal.normal[verticesIndex[i, 1], 1, :]
#             pfh_local = computePairFeatures(ps, pt, ns, nt)
#             pfh_local_list.append(pfh_local)
#             if j == ind.shape[1] - 1:
#                 pfhFeature.append(torch.cat(pfh_local_list))
#                 j = 1
#             else:
#                 j = j + 1
#         return torch.stack(pfhFeature)


def computePairFeatures(ps, pt, ns, nt):
    dptps = pt - ps
    dist = dptps.norm()
    ns_copy = copy.deepcopy(ns)
    nt_copy = copy.deepcopy(nt)
    angle1 = ns_copy.dot(dptps) / dist
    angle2 = nt_copy.dot(dptps) / dist

    if angle1.abs().acos() > angle2.abs().acos():
        ns_copy = nt
        nt_copy = ns
        dptps = -1 * dptps
        phi = -1 * angle2
    else:
        phi = angle1
    v = dptps.cross(ns_copy)
    v_norm = v.norm()
    if 0 == v_norm:
        return 0, 0, 0, 0
    else:
        v = v / v_norm
    w = ns_copy.cross(v)
    alpha = v.dot(nt_copy)
    theta = torch.atan2(w.dot(nt_copy), ns_copy.dot(nt_copy))
    # localPFH = [alpha, phi, theta, dist]
    localPF = torch.tensor([theta, alpha, phi, dist])
    return localPF


# s-->source, t-->target ps=[batch,N1,N2,dim], pt=[batch,N1,N2,dim]
# ns -->[batch,N1,N1,dim], and nt=[batch,N1,N2,dim]
def computeBatchPairFeatures(ps, pt, ns, nt):
    if True:
        dptps = pt - ps
        dist = dptps.norm(dim=3)
        ns_copy = ns.clone().detach()
        nt_copy = nt.clone().detach()
        # start = time.time()
        angle1 = bdot(ns_copy, dptps) / dist
        phi = angle1
        angle2 = bdot(nt_copy, dptps) / dist
        phi2 = angle2
        # end = time.time()
        # print("angel1: ", end - start)angle2 = bdot(nt_copy, dptps) / dist
        if False:
            angle2 = bdot(nt_copy, dptps) / dist

            # start = time.time()
            angleSelect = angle1.abs().acos() > angle2.abs().acos()
            # end = time.time()
            # print("angleSelect: ", end - start)
            ns_copy[angleSelect] = nt[angleSelect]
            nt_copy[angleSelect] = ns[angleSelect]
            dptps[angleSelect] = -1 * dptps[angleSelect]
            phi[angleSelect] = -1 * angle2[angleSelect]

        # start = time.time()

        # v = ns_copy.cross(dptps/dist.unsqueeze(dim=3))
        # end = time.time()
        # print("cross: ", end - start)
        v = dptps.cross(ns_copy)
        v_norm = v.norm(dim=3, keepdim=True)
        v = v / v_norm
        w = ns_copy.cross(v)

        alpha = bdot(v, nt_copy)
        # start = time.time()
        theta = torch.atan2(bdot(w, nt_copy), bdot(ns_copy, nt_copy))
        # end = time.time()
        # print("atan2: ", end - start)
        alpha = alpha.unsqueeze(dim=3)
        phi = phi.unsqueeze(dim=3)
        phi2 = phi2.unsqueeze(dim=3)
        theta = theta.unsqueeze(dim=3)
        # phi_normal = torch.atan2(torch.norm(ns_copy.cross(nt_copy), p=3), bdot(ns_copy, nt_copy))
        # phi_normal = phi_normal.unsqueeze(dim=3)
        dist = dist.unsqueeze(dim=3)
        localPF = torch.cat([theta, alpha, phi, phi2, dist], dim=3)  # B*N1*N2*4
        # localPF = torch.cat([theta, alpha, phi], dim=3)  # B*N1*N2*3
        return localPF
    else:
        dptps = pt - ps
        dist = torch.norm(dptps, dim=3, p=2, keepdim=True)
        theta = torch.atan2(torch.norm(ns.cross(dptps), p=3), bdot(ns, dptps))
        alpha = torch.atan2(torch.norm(nt.cross(dptps), p=3), bdot(nt, dptps))
        phi = torch.atan2(torch.norm(ns.cross(nt), p=3), bdot(ns, nt))

        phi = phi.unsqueeze(dim=3)
        theta = theta.unsqueeze(dim=3)
        alpha = alpha.unsqueeze(dim=3)
        localPF = torch.cat([theta, alpha, phi, dist], dim=3)  # B*N1*N2*4
        return localPF
    # def get_angle(v1, v2):
    #     return torch.atan2(torch.cross(v1, v2, dim=3).norm(p=2, dim=3), bdot(v1, v2))
    #
    # pseudo = ps - pt
    # dist = pseudo.norm(dim=3).unsqueeze(dim=3)
    # alpha = get_angle(ns, pseudo)
    # phi = get_angle(nt, pseudo)
    # theta = get_angle(ns, nt)
    # alpha = alpha.unsqueeze(dim=3)
    # phi = phi.unsqueeze(dim=3)
    # theta = theta.unsqueeze(dim=3)
    # localPF = torch.cat([alpha, phi, theta, dist], dim=3)  # B*N1*N2*4
    # print("new:", localPF[0, 0, index, :])
    # return localPF


def bdot(a, b):  # it has the form [b,N1,N2] and [b,N1,N2]
    # assert (a.dim() == b.dim())
    return (a * b).sum(dim=a.dim() - 1)


class BatchFeatureMAP:
    def __init__(self, points, maxKnn, includeSelf=False, queryPoints=None):  # points[batch_size, N, dim]

        self.points = points
        self.queryPoints = queryPoints
        self.maxKnn = maxKnn
        self.includeSelf = includeSelf
        self.__batchNormals = None
        self.__queryNormals = None
        self.knn_indices = None
        self.kdTree = BatchKnn(points, self.maxKnn, includeSelf, queryPoints=queryPoints)

    @property
    def batchNormal(self):
        return self.__batchNormals

    @batchNormal.setter
    def batchNormal(self, normals):
        self.__batchNormals = normals  # normals [batch,N,dim]

    @property
    def queryNormals(self):
        return self.__queryNormals

    @queryNormals.setter
    def queryNormals(self, normals):
        self.__queryNormals = normals  # normals [batch,N,dim]

    # def computeNormals(self):
    #     self.__batchNormals = EN.BatchEstimateNormals(self.points, self.maxKnn, self.includeSelf).normal

    def computeBatchSPF(self, queryIndex):
        ps, pt, ns, nt = self.findNN(queryIndex)
        simplifiedPointFeature = computeBatchPairFeatures(ps, pt, ns, nt)  # B*Nq*maxKnn*4
        if self.queryPoints is None:
            simplifiedPointFeature = simplifiedPointFeature.view(self.points.shape[0], queryIndex.size()[0], -1)
        else:
            simplifiedPointFeature = simplifiedPointFeature.view(self.queryPoints.shape[0], queryIndex.size()[0], -1)
        return simplifiedPointFeature

    def findNN(self, queryIndex):
        if self.queryPoints is not None:
            ps = self.queryPoints[:, queryIndex, :]
            ns = self.__queryNormals[:, queryIndex, :]
        else:
            ps = self.points[:, queryIndex, :]  # B*Nq*dim
            ns = self.__batchNormals[:, queryIndex, :]  # B*Nq*dim

        pt, indices = self.kdTree.queryIndics(queryIndex)  # pt --> B*Nq*maxKnn*dim
        self.knn_indices = indices
        nt = torch.stack([self.__batchNormals[i, indices[i, :], :] for i in range(self.__batchNormals.shape[0])])
        ps = ps.unsqueeze(dim=2).repeat(1, 1, self.maxKnn, 1)  # B*Nq*maxKnn*dim
        ns = ns.unsqueeze(dim=2).repeat(1, 1, self.maxKnn, 1)  # B*Nq*maxKnn*dim
        return ps, pt, ns, nt

    @staticmethod
    def createFullConnectGraph(index_):
        ind = torch.arange(0, index_).unsqueeze(dim=0)
        x1 = ind.repeat(ind.shape[1], 1).transpose(0, 1).reshape(1, -1)
        y1 = ind.repeat(ind.shape[1], 1).reshape(1, -1)
        xy = torch.cat((x1, y1), 0)
        tt = torch.arange(0, xy.shape[1], ind.shape[1] + 1)
        tt2 = torch.arange(0, xy.shape[1])
        tt3 = np.delete(np.asarray(tt2), np.asarray(tt))
        verticesIndex = xy[:, tt3]
        return verticesIndex

    # def computeBatchPF_full(self):
    #     queryIndex = torch.arange(self.points.shape[1])
    #     ps_, pt_, ns_, nt_ = self.findNN(queryIndex)  # B*Nq*maxKnn*dim
    #     verticesIndex = BatchFeatureMAP.createFullConnectGraph(self.maxKnn)  # 2*N1
    # pair_vertices = self.points[:, verticesIndex, :]  # B*2*N2*dim
    # pair_vertices_normal = self.batchNormals[:, verticesIndex, :]  # B*2*N2*dim
    # ps = pair_vertices[:, 0, :, :].unsqueeze(dim=1)
    # pt = pair_vertices[:, 1, :, :].unsqueeze(dim=1)
    # ns = pair_vertices_normal[:, 0, :, :].unsqueeze(dim=1)
    # nt = pair_vertices_normal[:, 1, :, :].unsqueeze(dim=1)
    # ps = ps_[:, :, verticesIndex[0, :], :]
    # pt = pt_[:, :, verticesIndex[1, :], :]
    # ns = ns_[:, :, verticesIndex[0, :], :]
    # nt = nt_[:, :, verticesIndex[1, :], :]
    #
    # fpf = computeBatchPairFeatures(ps, pt, ns, nt)
    # # return B*1*N2*4, where N2=self.points.shape[2]*(self.points.shape[2]-1)
    # # TODO put the shape together
    # fpd = fpf.view(self.points.shape[0], self.points.shape[1], -1)
    # return fpd

    @staticmethod
    def computePF(points, normals):
        # points=[b,N,k,dim], normals is [b,N,k,dim]
        # print("points: ", points.shape)
        verticesIndex = BatchFeatureMAP.createFullConnectGraph(points.shape[2])  # 2*N2 --> N2=k*k-1
        # print("vertices: size: ", verticesIndex.shape)
        ps = points[:, :, verticesIndex[0, :], :]
        pt = points[:, :, verticesIndex[1, :], :]
        ns = normals[:, :, verticesIndex[0, :], :]
        nt = normals[:, :, verticesIndex[1, :], :]
        # start = time.time()
        fpf = computeBatchPairFeatures(ps, pt, ns, nt)  # B*N*N2*4
        # end = time.time()
        # print("meaure compute batch pp: ", end - start)
        fpd = fpf.view(points.shape[0], points.shape[1], points.shape[2], -1)
        return fpd


if __name__ == '__main__':
    # train_loader = DataLoader(
    #    ModelNet40SphericalHarmonics(maxDegree=6, partition='train'), num_workers=8, batch_size=6, shuffle=False,
    #    drop_last=True)
    # for dataset in train_loader:
    #    shData, points, normal, label = dataset
    #    bfmap = BatchFeatureMAP(points, 10, False)
    #    bfmap.batchNormal = normal
    #    index = torch.arange(points.shape[1])
    #    start = time.time()
    #    bfmap.computeBatchSPF(index)
    #    end = time.time()
    #    print("SPF: ", end - start)
    #    break
    # bfmap = BatchFeatureMAP(x, 5, False)
    # bfmap.computeNormals()
    # start = time.time()
    # index = torch.arange(x.shape[1])
    # bfmap.computeBatchSPF(index)
    # end = time.time()
    # print("SPF: ", end - start)
    # start = time.time()
    # y = torch.rand(8, 100, 10, 3)
    # n = torch.rand(8, 100, 10, 3)
    # BatchFeatureMAP.computePF(y, n)
    # end = time.time()
    # print("FPF: ", end - start)
    ps = torch.tensor([-0.0456, -0.1121, -0.1379])
    pt = torch.tensor([-0.0754, -0.1061, -0.1048])
    ns = torch.tensor([-0.2338, -0.9717, 0.0345])
    nt = torch.tensor([0.4487, 0.8931, -0.0302])
    print(computePairFeatures(ps, pt, ns, nt))
