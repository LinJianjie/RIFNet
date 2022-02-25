import open3d as o3d
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from basic.common import TorchClusteringKNN, BatchKnn
import numpy as np

import time


class EstimateNormals:
    def __init__(self, points, maxKnn, includeSelf=True):
        self.points = points
        self.maxKnn = maxKnn
        self.kdTree = TorchClusteringKNN(points, self.maxKnn, includeSelf)
        self.normal = torch.zeros(points.shape[0], 3)
        for i in range(points.shape[0]):
            self.normal[i, :] = self.computeNormal(i)

    def computeNormal(self, index):
        knnPoints = self.points[self.kdTree.query(index)[0, :], :]
        covariance = torch.zeros([3, 3])
        cumulants = torch.zeros(9, 1)
        for point in knnPoints:
            cumulants[0] += point[0]
            cumulants[1] += point[1]
            cumulants[2] += point[2]
            cumulants[3] += point[0] * point[0]
            cumulants[4] += point[0] * point[1]
            cumulants[5] += point[0] * point[2]
            cumulants[6] += point[1] * point[1]
            cumulants[7] += point[1] * point[2]
            cumulants[8] += point[2] * point[2]
        cumulants /= knnPoints.shape[0]
        covariance[0, 0] = cumulants[3] - cumulants[0] * cumulants[0]
        covariance[1, 1] = cumulants[6] - cumulants[1] * cumulants[1]
        covariance[2, 2] = cumulants[8] - cumulants[2] * cumulants[2]
        covariance[0, 1] = cumulants[4] - cumulants[0] * cumulants[1]
        covariance[1, 0] = covariance[0, 1]
        covariance[0, 2] = cumulants[5] - cumulants[0] * cumulants[2]
        covariance[2, 0] = covariance[0, 2]
        covariance[1, 2] = cumulants[7] - cumulants[1] * cumulants[2]
        covariance[2, 1] = covariance[1, 2]
        eigenvalues, eigenvector = torch.eig(covariance, True)
        return eigenvector[:, 0]


class BatchEstimateNormalsOld:
    def __init__(self, points, maxKnn, includeSelf=True):
        self.points = points
        self.maxKnn = maxKnn
        self.kdTree = BatchKnn(points, self.maxKnn, includeSelf)
        self.normal = self.computeNormal()[:, :, :, 0]
        # self.normal = torch.zeros(points.shape[0], points.shape[1], 3)
        # for i in range(points.shape[1]):
        #     self.normal[:, i, :] = self.computeNormal(i)

    def computeNormal(self):
        # clusters = self.kdTree.query(index)
        # covariance = torch.zeros([clusters.shape[0], 3, 3])
        # cumulants = torch.zeros(clusters.shape[0], 9)
        # for i in range(clusters.shape[1]):
        #     cluster = clusters[:, i, :]
        #     cumulants[:, 0:3] += cluster[:, :]
        #     cumulants[:, 3:6] += cluster[:, 0].unsqueeze(1) * cluster[:, :]
        #     cumulants[:, 6:8] += cluster[:, 1].unsqueeze(1) * cluster[:, 1:]
        #     cumulants[:, 8] += cluster[:, 2] * cluster[:, 2]
        # cumulants /= clusters.shape[1]
        # covariance[:, 0, 0] = cumulants[:, 3] - cumulants[:, 0] * cumulants[:, 0]
        # covariance[:, 1, 1] = cumulants[:, 6] - cumulants[:, 1] * cumulants[:, 1]
        # covariance[:, 2, 2] = cumulants[:, 8] - cumulants[:, 2] * cumulants[:, 2]
        # covariance[:, 0, 1] = cumulants[:, 4] - cumulants[:, 0] * cumulants[:, 1]
        # covariance[:, 1, 0] = covariance[:, 0, 1]
        # covariance[:, 0, 2] = cumulants[:, 5] - cumulants[:, 0] * cumulants[:, 2]
        # covariance[:, 2, 0] = covariance[:, 0, 2]
        # covariance[:, 1, 2] = cumulants[:, 7] - cumulants[:, 1] * cumulants[:, 2]
        # covariance[:, 2, 1] = covariance[:, 1, 2]
        # eigenVec = torch.zeros(clusters.shape[0], 3)
        # e, v = torch.symeig(covariance, True)
        # eigenVec = v[:, 0]

        # for i in range(clusters.shape[0]):
        #     eigenvalues, eigenvector = torch.eig(covariance[i, :], True)
        #     eigenVec[i, :] = eigenvector[:, 0]
        # return eigenVec
        index = torch.arange(self.points.shape[1])
        clusters = self.kdTree.query(index)
        covariance = torch.zeros([clusters.shape[0], clusters.shape[1], 3, 3])
        cumulants = torch.zeros(clusters.shape[0], clusters.shape[1], 9)
        cumulants[:, :, 0:3] = torch.sum(clusters, dim=2)
        cumulants[:, :, 3:6] = torch.sum(clusters[:, :, :, 0].unsqueeze(3) * clusters, dim=2)
        cumulants[:, :, 6:8] = torch.sum(clusters[:, :, :, 1].unsqueeze(3) * clusters[:, :, :, 1:], dim=2)
        cumulants[:, :, 8] = torch.sum(clusters[:, :, :, 2] * clusters[:, :, :, 2], dim=2)
        cumulants /= float(self.maxKnn)
        covariance[:, :, 0, 0] = cumulants[:, :, 3] - cumulants[:, :, 0] * cumulants[:, :, 0]
        covariance[:, :, 1, 1] = cumulants[:, :, 6] - cumulants[:, :, 1] * cumulants[:, :, 1]
        covariance[:, :, 2, 2] = cumulants[:, :, 8] - cumulants[:, :, 2] * cumulants[:, :, 2]
        covariance[:, :, 0, 1] = cumulants[:, :, 4] - cumulants[:, :, 0] * cumulants[:, :, 1]
        covariance[:, :, 1, 0] = covariance[:, :, 0, 1]
        covariance[:, :, 0, 2] = cumulants[:, :, 5] - cumulants[:, :, 0] * cumulants[:, :, 2]
        covariance[:, :, 2, 0] = covariance[:, :, 0, 2]
        covariance[:, :, 1, 2] = cumulants[:, :, 7] - cumulants[:, :, 1] * cumulants[:, :, 2]
        covariance[:, :, 2, 1] = covariance[:, :, 1, 2]
        e, v = torch.symeig(covariance, True)
        return v


class BatchEstimateNormals:
    def __init__(self, points_, maxKnn, includeSelf=True):  # [batch,N,3]
        points = points_.numpy()
        self.normal = torch.zeros_like(points_)
        for i in range(points.shape[0]):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[i, :, :])
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=maxKnn))
            self.normal[i, :, :] = torch.from_numpy(np.asarray(pcd.normals))


def computeFPFH(points_, normals_):
    points = points_.numpy()
    normals = normals_.numpy()
    if points.ndim == 3:
        features = torch.zeros(points.shape[0], points.shape[1], 33)
        for i in range(points.shape[0]):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[i, :, :])
            pcd.normals = o3d.utility.Vector3dVector(normals[i, :, :])
            f1 = o3d.registration.compute_fpfh_feature(pcd,
                                                       o3d.geometry.KDTreeSearchParamKNN(knn=20))
            features[i, :, :] = torch.from_numpy(np.asarray(f1.data).transpose())
        return features
    if points.ndim == 4:
        features = torch.zeros(points.shape[0], points.shape[1], points.shape[2], 33)
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[i, j, :, :])
                pcd.normals = o3d.utility.Vector3dVector(normals[i, j, :, :])
                f1 = o3d.registration.compute_fpfh_feature(pcd,
                                                           o3d.geometry.KDTreeSearchParamKNN(knn=20))
                features[i, j, :, :] = torch.from_numpy(np.asarray(f1.data).transpose())
        return features


if __name__ == '__main__':
    x = torch.rand(8, 1024, 3)
    start = time.time()
    ben = BatchEstimateNormals(x, 10, False)
