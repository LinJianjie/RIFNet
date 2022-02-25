import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from basic.EstimateNormals import BatchEstimateNormals, computeFPFH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from basic.PointFeatureMap import BatchFeatureMAP
from basic.constant import *
from torch.autograd import Variable
import numpy as np
from basic.common import minPointNet
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from basic.sphspf_utils import *
import time

"""
a set abstraction level consists of sampling layer, grouping layer, and pointnet layer
PoinetNet:
Given: input is a matrix with N X (d+C) --> N points, each point has d-dim coordinates(e.g. x,y,z) and C-dim point feature
    Output: N' X (d+C') --> N' subsampled points with d-dim coordinates and new C' dim feature vectors summarizing local context

1. the Sampling Layers select a set of points from input points (from the original point cloud) --> define the centroids of local regions
    a. given points {x1,x2,...,xn},  using farthest point sampling fps --> get {xi_1, xi_2,..., xi_N'} such that xi_j is 
        the most distant point from the  {xi_1, xi_2,..., xi_N'} --> therefore the number of Points from N is reduced to 
        N'
    
2. Grouping Layer constructs local region sets by finding neihboring points around the centroids
    a. the input has size of N X (d+C), after Sampling Layer, we got N' centroid with d dim
    b. search each centroid from Sampling Layer, doing the ball search with maximum K value 
    c. therefore the out is N' x K x (d+C)
3. PointNet Layer uses a mini-PointNet to encode local region patterns into features vector
    a. the input for pointnet Layer are N' local regions of points with data size N' x k x (d+C)
    b. for each point inside the local regions, doing the local transformation which is indicates as x^j_i=x^j_i-hat(x)^j
    c. the output will become N' x (d+C')

adaption to our:
Given: the input with N x C
output: N'x C'
1. Augmentation Layer: Nx(SH+C+d), where SH indicates the spherical harmonics feature, and d is the N point with d-dim, and C is the features from the previous layer
2. Sampling Layer: select N' centroid from N using FPS
3. Grouping Layer: for each Point from N' using ball search with maximum K, results in N' x K x (SH+C+d)
4. FPH Layer: inside the one Group or one Local region, we doing the FPH as a features result in d'
5. Pointnet Layer: for each Kx(SH+C+d') result in a C' feature vector, therefore we have output N'x C', 
    where N' is the centroid from the result of FPS

"""


class Set_Abstraction_Module(torch.nn.Module):
    def __init__(self, ratio, in_channel, mlp, K, use_fpfh=False, maxdegree=20, use_all_spf=True,
                 out_channel_transformation3D=64):
        """
        Args:
            ratio: for compute the N'
            mlp: for the output channel
            in_channel: is the size of featurvector
            K: for knn
        """
        super(Set_Abstraction_Module, self).__init__()
        self.ratio = ratio  # which is used to determine N' value, N'=N*ratio
        self.K = K  # the knn
        self.use_fpfh = use_fpfh
        self.use_all_spf = use_all_spf
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.in_channel = in_channel
        self.max_degree = maxdegree
        self.pose_cloud = None
        self.pose_normal = None
        if self.use_fpfh:
            self.rri = featureTransformNet(self.max_degree)
            last_channel = in_channel + out_channel_transformation3D  # TODO check
        else:
            self.rri = featureTransformation3D(self.max_degree,
                                               mlp=[64, out_channel_transformation3D],
                                               in_channel=FPH_DIM + 1,
                                               out_channel=out_channel_transformation3D)  # B,N,64
            last_channel = in_channel + out_channel_transformation3D  # TODO check
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pointCloudPose, featureVector, PointCloudNormal, SH, fpfh=None):
        # idx is the select centroid
        # tha featureVector has size B*N*C
        # pointClouxPose --> has Size B*N*3
        """
        Args:
            N: the number of points
            C: the size of each feature
            B: the Batch Size
            pointCloudPose: (B,N,3)
            PointCloudNormal:(B,N,3)
            featureVector: (B,N,C)
            minKNN:

        Returns:
            new_pointCloudPose:(B,N',3)
            new_featureVector: (B,N,C')

        """
        # sampling Layer
        new_PointCloudPose, new_PointCloud_Normal, pose_cluster, feature_cluster, normal_cluster, \
        sh_cluster, new_SHFeature, fpfh_cluster, new_FPFH = \
            self.samplingLayer(pointCloudPose, featureVector, PointCloudNormal, SH, fpfh)
        # group Layer
        grouping_feature = self.groupLayer(pose_cluster, feature_cluster, normal_cluster, sh_cluster, fpfh_cluster)
        # minPointNet
        new_featureVector = self.minPointNetLayer(grouping_feature)
        return new_PointCloudPose, new_featureVector, new_PointCloud_Normal, new_SHFeature, new_FPFH
        # Concatenates sequence of tensors along a new dimens

    def samplingLayer(self, pointCloudPose, featureVector, pointCloudNormal, SH_Features=None, FPFHFeature=None):
        batch_size, N, dim = pointCloudPose.shape
        self.pose_cloud = pointCloudPose
        self.pose_normal = pointCloudNormal
        pose_batch = pointCloudPose.reshape(-1, dim)
        if featureVector is not None:
            feature_batch = featureVector.reshape(-1, featureVector.shape[2])
        else:
            feature_batch = None
        normal_batch = pointCloudNormal.reshape(-1, pointCloudNormal.shape[2])
        if SH_Features is not None:
            sh_batch = SH_Features.reshape(-1, SH_Features.shape[2])
        if FPFHFeature is not None:
            fpfh_batch = FPFHFeature.reshape(-1, FPFHFeature.shape[2])
        batch_index = torch.arange(batch_size, device=pointCloudPose.device)
        #batch_index = torch.arange(batch_size).cuda()
        batch_fps = batch_index.repeat(N, 1).transpose(0, 1).reshape(-1)
        fps_idx = torch_geometric.nn.fps(pose_batch, batch_fps, self.ratio)
        centroid_size = int(fps_idx.shape[0] / batch_size)

        new_PointCloudPose = pose_batch[fps_idx].reshape(batch_size, -1, dim)
        new_PointCloudNormal = normal_batch[fps_idx].reshape(batch_size, -1, dim)
        if SH_Features is not None:
            new_SHFeature = sh_batch[fps_idx].reshape(batch_size, -1, SH_Features.shape[2])
        else:
            new_SHFeature = None
        if FPFHFeature is not None:
            new_FPFHFeature = fpfh_batch[fps_idx].reshape(batch_size, -1, FPFHFeature.shape[2])
        else:
            new_FPFHFeature = None
        # start=time.time()
        knn_index = torch_geometric.nn.knn(pose_batch, pose_batch[fps_idx],
                                           self.K, batch_fps, batch_fps[fps_idx])
        # end=time.time()
        # print("knn: ",end-start)
        local_pose_cluster = pose_batch[knn_index[1, :], :]
        if featureVector is not None:
            local_feature_cluster = feature_batch[knn_index[1, :], :]
        else:
            local_feature_cluster = None
        local_normal_cluster = normal_batch[knn_index[1, :], :]
        if SH_Features is not None:
            local_sh_cluster = sh_batch[knn_index[1, :], :]
        if FPFHFeature is not None:
            local_fpfh_cluster = fpfh_batch[knn_index[1, :], :]

        pose_cluster = local_pose_cluster.reshape(batch_size, centroid_size, self.K, dim)  # B*N'*minKNN*d,
        if featureVector is not None:
            feature_cluster = local_feature_cluster.reshape(batch_size, centroid_size, self.K, featureVector.shape[2])
        else:
            feature_cluster = None
        normal_cluster = local_normal_cluster.reshape(batch_size, centroid_size, self.K, dim)
        if SH_Features is not None:
            sh_cluster = local_sh_cluster.reshape(batch_size, centroid_size, self.K, SH_Features.shape[2])
        else:
            sh_cluster = None
        if FPFHFeature is not None:
            fpfh_cluster = local_fpfh_cluster.reshape(batch_size, centroid_size, self.K, FPFHFeature.shape[2])
        else:
            fpfh_cluster = None

        return new_PointCloudPose, new_PointCloudNormal, pose_cluster, feature_cluster, normal_cluster, sh_cluster, new_SHFeature, fpfh_cluster, new_FPFHFeature

    def groupLayer(self, pose_cluster, feature_cluster, normal_cluster, sh_cluster, fpfh_cluster=None):
        # AllBatchLocalRegionFeatures = []
        # for atBatch in range(pointCloudCluster.shape[0]):  # at the batch
        #     oneBatchLocalRegionFeatures = []
        #     for atCenter in range(pointCloudCluster.shape[1]):  # at the k_th cluster
        #         FPHFeature = PointFeatureMap(pointCloudCluster[atBatch, atCenter, :, :], self.K).computePFH()
        #         feature = torch.cat(
        #             [FPHFeature, feature_cluster[atBatch, atCenter, :, :], pointCloudCluster[atBatch, atCenter, :, :]],
        #             dim=1)
        #         oneBatchLocalRegionFeatures.append(feature)
        #     AllBatchLocalRegionFeatures.append(torch.stack(oneBatchLocalRegionFeatures))
        # groupFeature = torch.stack(AllBatchLocalRegionFeatures)  # B*N'*K*(d+c+f), where f=4
        if self.use_fpfh:
            # point_feature = computeFPFH(pose_cluster.cpu(), normal_cluster.cpu()).to(pose_cluster.device)
            point_feature = fpfh_cluster

        else:
            if self.use_all_spf:
                queryPoint = pose_cluster.reshape(pose_cluster.shape[0], -1, 3)
                queryPointNormal = normal_cluster.reshape(normal_cluster.shape[0], -1, 3)
                bfmap = BatchFeatureMAP(self.pose_cloud.clone().detach(), self.K, False, queryPoints=queryPoint)
                bfmap.batchNormal = self.pose_normal
                bfmap.queryNormals = queryPointNormal
                index = torch.arange(queryPoint.shape[1])
                point_feature = bfmap.computeBatchSPF(index)
                point_feature = point_feature.reshape(pose_cluster.shape[0], pose_cluster.shape[1],
                                                      pose_cluster.shape[2], -1)
            else:
                point_feature = BatchFeatureMAP.computePF(pose_cluster, normal_cluster)  # B*N'*K*(c)

        # pose_dis = pose_cluster.clone().detach()
        # pose_dis = torch.norm(pose_dis, dim=3, p=2, keepdim=True)
        pose_dis = distance(pose_cluster)

        if self.use_fpfh:
            pass
        else:
            # feature_rri = self.rri(point_feature, torch.cat([sh_cluster, pose_dis], dim=3))
            feature_rri = self.rri(point_feature, pose_dis)
            # feature_rri = self.rri(point_feature, None)
        if feature_cluster is not None:
            grouping_feature = torch.cat([feature_cluster, feature_rri], dim=3)
        else:
            grouping_feature = feature_rri
        return grouping_feature

    def minPointNetLayer(self, groupFeature):
        new_featureVector = groupFeature.permute(0, 3, 2, 1)  # B*(d+c+f)*K*N', where f=(self.maxKnn-1)*4
        for i, conv2d in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_featureVector = F.relu(bn(conv2d(new_featureVector)))
        new_featureVector = torch.max(new_featureVector, 2)[0].permute(0, 2, 1)
        return new_featureVector


class GlobalSAModule(torch.nn.Module):
    def __init__(self, in_channel, mlp):
        super(GlobalSAModule, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dim = PC_DIM
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        # new_x = torch.cat([pointCloudPose, featureVector], dim=2)
        new_featureVector = x.permute(0, 2, 1)  # B*(d+c)*K*N'
        for i, conv1d in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_featureVector = F.relu(bn(conv1d(new_featureVector)))
        new_featureVector = torch.max(new_featureVector, dim=-1)[0]
        return new_featureVector


class SHTransformation(nn.Module):
    def __init__(self, maxdegree):
        super(SHTransformation, self).__init__()
        self.maxDegree = maxdegree
        self.out_channel = (maxdegree + 1) ** 2
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.out_channel)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = torch.max(x1, 2, keepdim=True)[0]
        x1 = x1.view(-1, 1024)
        x1 = F.relu(self.bn4(self.fc1(x1)))
        x1 = F.relu(self.bn5(self.fc2(x1)))
        x1 = self.fc3(x1)
        # x1 = self.bn1(self.conv1(x))
        # x1 = self.bn3(self.conv3(x1))
        # x1 = torch.mean(x1, 2, keepdim=True)
        x1 = x1.view(x.shape[0], 1, self.out_channel)
        x1 = x1.repeat(1, x.shape[2], 1)
        return x1

    def getSHRIFFeature(self, anm, Ynm, maxSHDegree):
        if Ynm.shape[0] != anm.shape[0] or Ynm.shape[1] != anm.shape[1] or Ynm.shape[2] != anm.shape[2]:
            raise ValueError("")
        reconstruction = anm * Ynm
        currIter = 0
        shFeature = torch.zeros([Ynm.shape[0], Ynm.shape[1], self.maxDegree + 1]).to(Ynm.device)
        #shFeature = torch.zeros([Ynm.shape[0], Ynm.shape[1], self.maxDegree + 1]).cuda()
        for i in range(self.maxDegree + 1):
            temp = reconstruction[:, :, range(currIter, currIter + 2 * i + 1)]
            shFeature[:, :, i] = torch.norm(temp, p=2, dim=2, keepdim=False)
            currIter += 2 * i + 1
        return shFeature


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
            #iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class RRI(nn.Module):

    def __init__(self):
        super(RRI, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, pose):
        y = []
        B, N, D = x.shape
        # for i in range(int(D / 4)):
        #     x1 = x[:, :, 4 * i:4 * (i + 1)]
        #     x1 = x1.transpose(2, 1)
        #     x2 = F.relu(self.bn1(self.conv1(x1)))
        #     x2 = x2.unsqueeze(dim=3)
        #     y.append(x2)
        # maxpol = torch.max(x2, 1, keepdim=True)[0]
        # y.append(maxpol)
        x1 = torch.split(x, 4, dim=2)
        x1 = torch.stack(x1, dim=3)  # B,N,4, k
        x2 = pose.unsqueeze(dim=3)  # B,N,1, 1
        x2 = x2.repeat(1, 1, 1, x1.shape[3])  # B,N,1,K
        x3 = torch.cat([x1, x2], dim=2).permute(0, 2, 1, 3)  # B,N,1+4,K --> B,5,N,K
        x3 = self.conv1(x3)  # B,64,N,K
        x3 = self.conv2(x3)  # B,64,N,K
        x3 = x3.max(dim=-1, keepdim=False)[0]  # B,64,N
        x3 = x3.transpose(2, 1)
        return x3


class STNkd(nn.Module):
    def __init__(self, k=128):
        super(STNkd, self).__init__()
        mlp = [128, 512]
        self.conv1 = torch.nn.Conv1d(k, mlp[0], 1)
        self.conv2 = torch.nn.Conv1d(mlp[0], mlp[1], 1)
        self.conv3 = torch.nn.Conv1d(mlp[1], 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.bn2 = nn.BatchNorm1d(mlp[1])
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
            #iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def MLP(channels, batch_norm=True):
    return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
                 for i in range(1, len(channels))
                 ])


def pca_preprocessing(points, usespf):
    pca = torch.pca_lowrank(points)
    pcd_coordiate = pca[2]
    new_points = torch.bmm(points, pcd_coordiate)
    if usespf:
        normals = BatchEstimateNormals(new_points.cpu(), 10, False).normal
        return new_points, normals.to(points.device)
        #return new_points, normals.cuda()
    else:
        return new_points, None


class PointNetBasic(nn.Module):
    def __init__(self, in_channels, mlp, K, maxDegree=8):
        super(PointNetBasic, self).__init__()
        self.K = K
        self.maxDegree = maxDegree
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.useSH = True
        self.useSTN = False
        self.useFeatureTransformation = False
        self.useFeature = ~self.useSH and False
        self.use_pca_preprocessing = False
        self.use_dynamic = True

        self.use_sh_transformation = False
        self.useSH_feature = False

        self.use_pose = True
        self.use_pose_dist = False

        self.use_spf = False
        self.use_rri = False
        if self.use_pose and self.use_pose_dist:
            raise ValueError(" pose and  pose distance should only one of them be used")
        if self.useSH_feature and self.use_sh_transformation:
            raise ValueError("useSH_feature and  Sh transforamtion should only one of them be used")
        if self.useSH:
            # last_channel = in_channels + FPH_DIM * self.K
            last_channel = 0
            if self.useSH_feature or self.use_sh_transformation:
                last_channel = last_channel + 9
            if self.use_spf:
                last_channel = last_channel + FPH_DIM * self.K
            if self.use_pose:
                last_channel = last_channel + 3
            if self.use_pose_dist:
                last_channel = last_channel + 1

            if self.useSTN:
                self.stn = STN3d(3)
            if self.useFeatureTransformation:
                self.fstn = STNkd(k=128)
            if self.use_rri:
                self.rri = RRI()
                if self.use_pose:
                    last_channel = 64 + 3
                elif self.use_pose_dist:
                    last_channel = 64
                else:
                    last_channel = 64
                # if ~self.use_pose and ~self.use_pose_dist:
                #    last_channel = 64
            if self.use_sh_transformation:
                self.sphTN = SHTransformation(self.maxDegree)
            if self.use_dynamic:
                self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k=20,
                                             aggr="max")  # without spf will be set 3 otherwise it will be set 64
                self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128]), k=20, aggr="max")
                last_channel = 128 + 64
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel
        else:
            last_channel = 3
            if self.useFeature:
                self.stn = STN3d(last_channel)
                self.fstn = STNkd(k=64)
            mlp = [64, 128, 256, 1024]
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel
        # self.conv2 = DynamicEdgeConv(minPointNet([2 * 64, 128]), 20, "max")
        # self.lin1 = minPointNet([128 + 64, 1024])

    def forward(self, pose, SH, normal, Ynm=None):
        if self.useSH:
            if self.use_pca_preprocessing:
                pose, normal = pca_preprocessing(pose, self.use_spf)
            if self.use_spf:
                bfmap = BatchFeatureMAP(pose.clone().detach(), self.K, False)
                bfmap.batchNormal = normal
                index = torch.arange(pose.shape[1])
                simplifiedPointFeature = bfmap.computeBatchSPF(index)
            new_x = []
            if self.use_pose_dist:
                pose_dis = pose.clone().detach()
                pose_dis = torch.norm(pose_dis, dim=2, p=2, keepdim=True)

            # rri_input = simplifiedPointFeature.transpose(2, 1)
            # rri_output = self.rri(rri_input)
            # new_x = SH
            # new_x = torch.cat([pose, simplifiedPointFeature], dim=2)
            # new_x = SH
            if self.use_sh_transformation:
                SH_x = self.sphTN(pose.transpose(2, 1))
                SH_y = self.sphTN.getSHRIFFeature(SH_x, Ynm, self.maxDegree)
            if self.useSTN:
                x = pose.clone().detach()  # B,N,D
                x = x.transpose(2, 1)  # B,D,N
                trans = self.stn(x)
                x = x.transpose(2, 1)  # B,N,D
                x = torch.bmm(x, trans)
                if self.use_sh_transformation:
                    new_x = torch.cat([x, SH_y, simplifiedPointFeature], dim=2)
                elif self.useFeature:
                    new_x = torch.cat([x, SH, simplifiedPointFeature], dim=2)
                else:
                    new_x = torch.cat([x, simplifiedPointFeature], dim=2)
            else:
                if self.use_sh_transformation:
                    # new_x = torch.cat([pose, SH_y, simplifiedPointFeature], dim=2)
                    new_x.append(SH_y)
                if self.useSH_feature:
                    # new_x = torch.cat([pose, SH, simplifiedPointFeature], dim=2)
                    new_x.append([SH])
                if self.use_pose:
                    # new_x = torch.cat([pose, simplifiedPointFeature], dim=2)
                    new_x.append([pose])
                if self.use_spf:
                    # new_x = torch.cat([y, simplifiedPointFeature], dim=2)
                    new_x.append(simplifiedPointFeature)
                # if self.use_pose_dist:
                # new_x = torch.cat([pose, simplifiedPointFeature], dim=2)
                #    new_x.append(pose_dis)
                # instead using the x,y,z directly, we use the distance to the origil as radial profile
                if self.use_rri:
                    new_x = self.rri(simplifiedPointFeature, pose_dis)
                else:
                    # new_x = torch.cat(new_x, dim=2)
                    new_x = pose
            # new_x = SH
            # x = new_x.permute(0, 2, 1)  # batch *(maxDegree+1+3+4*(k-1))*N
            if self.use_dynamic:
                x = new_x.reshape(-1, new_x.shape[2])
                batchIndex = torch.arange(new_x.shape[0])
                batch = batchIndex.repeat(new_x.shape[1], 1).transpose(0, 1).reshape(-1).to(pose.device)
                #batch = batchIndex.repeat(new_x.shape[1], 1).transpose(0, 1).reshape(-1).cuda()
                x1 = self.conv1(x, batch)
                x1_reshape = x1.reshape(new_x.shape[0], new_x.shape[1], -1)
                x3 = self.conv3(x1, batch)
                x3_reshape = x3.reshape(new_x.shape[0], new_x.shape[1], -1)
                x = torch.cat([x1_reshape, x3_reshape], dim=2)  # with spf will add new_x otherwise without new_x
                x = x.transpose(2, 1)
            else:
                x = new_x.transpose(2, 1)
            #            x2=self.conv2(x1)
            #            out=self.lin1(torch.cat([x1,x2]),dim=1)
            #            out=global_max_pool(out)
            for i, conv1d in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                x = F.relu(bn(conv1d(x)))
                if i == 0 and self.useFeatureTransformation:
                    trans_feat = self.fstn(x)
                    x = x.transpose(2, 1)
                    x = torch.bmm(x, trans_feat)
                    x = x.transpose(2, 1)
        else:
            x = pose  # B,N,D
            if self.use_pca_preprocessing:
                pose, normal = pca_preprocessing(pose, False)
            if self.useSTN:
                x = x.transpose(2, 1)  # B,D,N
                trans = self.stn(x)
                x = x.transpose(2, 1)  # B,n
                x = torch.bmm(x, trans)
            x = x.transpose(2, 1)  # B,n
            for i, conv1d in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                x = F.relu(bn(conv1d(x)))
                if i == 0 and self.useFeatureTransformation:
                    trans_feat = self.fstn(x)
                    x = x.transpose(2, 1)
                    x = torch.bmm(x, trans_feat)
                    x = x.transpose(2, 1)
        return pose, x, normal


"""
The whole work process is 
Given Nxd points 
1. compute Spherical harmonics for each Points
2. compute the SFPH for each Point with K-nn
3. with result in N x (SH+Kx4) features
4. extracting high lever feature with point-net result in NxC features --> layer1
            
5. NxC --> |set_abstraction| --> N'xC' --> |set_abstraction| --> N''x C'' --> |set_abstraction| -->N'''x C''' --> Classification Layers 
        
"""
if __name__ == '__main__':
    pointCloudPose = torch.rand(8, 1024, 3)
    PointCloudNormal = torch.rand(8, 1024, 3)
    featureVector = torch.rand(8, 1024, 64)
    sa1_module = Set_Abstraction_Module(ratio=0.5, in_channel=64 + 3, mlp=[64, 64, 128], K=10)
    sa1_module(pointCloudPose, featureVector, PointCloudNormal)
    feature = torch.rand(8, 1024, 256)
    globalSa_module = GlobalSAModule(in_channel=256 + 3, mlp=[256, 512, 1024])
    globalSa_module(pointCloudPose, feature)
    pointBased = PointNetBasic(in_channels=8 + 3, mlp=[64, 64, 64], K=10)
    SH = torch.rand(8, 1024, 8)
    pointBased(pointCloudPose, SH, PointCloudNormal)
