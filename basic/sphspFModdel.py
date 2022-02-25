import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from basic.EstimateNormals import computeFPFH
from basic.PointFeatureMap import BatchFeatureMAP
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
import basic.SetAbstraction as Abstraction
from basic.sphspf_utils import *


class SPHSPFModel(nn.Module):
    def __init__(self, maxDegree):
        super(SPHSPFModel, self).__init__()
        self.K = 15
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.max_degree = maxDegree
        self.useFPFH = False
        self.use_feature = True
        self.use_global_rri = True
        self.use_all_spf = False
        self.use_GAT = False
        self.use_GAT_GRRI = False
        if self.use_feature:
            if self.useFPFH:
                self.rri = featureTransformNet(self.max_degree)
                last_channel = 64
            else:
                last_channel = 64
                self.rri = featureTransformation(in_channel=FPH_DIM + self.max_degree + 1 + 1,
                                                 mlp=[64, last_channel],
                                                 out_channel=last_channel,
                                                 use_gat=self.use_GAT)  # B,N,64
        else:
            last_channel = 0
        self.sa1_module = Abstraction.Set_Abstraction_Module(ratio=0.5,
                                                             in_channel=last_channel,
                                                             mlp=[128, 128, 256],
                                                             K=15,
                                                             use_fpfh=self.useFPFH,
                                                             maxdegree=self.max_degree,
                                                             use_all_spf=self.use_all_spf,
                                                             out_channel_transformation3D=64)

        self.sa2_module = Abstraction.Set_Abstraction_Module(ratio=0.25,
                                                             in_channel=256,
                                                             mlp=[256, 256, 512],
                                                             K=15,
                                                             use_fpfh=self.useFPFH,
                                                             maxdegree=self.max_degree,
                                                             use_all_spf=self.use_all_spf,
                                                             out_channel_transformation3D=64)
        if self.use_global_rri:
            out_channel = 64
            self.global_rri = featureTransformation(in_channel=FPH_DIM + self.max_degree + 1 + 1,
                                                    mlp=[64, 64, out_channel],
                                                    out_channel=out_channel,
                                                    use_gat=self.use_GAT_GRRI)
            self.globalSa_module = Abstraction.GlobalSAModule(in_channel=512 + out_channel, mlp=[512, 512, NUM_PC_Train])
        else:
            self.globalSa_module = Abstraction.GlobalSAModule(in_channel=512, mlp=[512, 512, NUM_PC_Train])

        self.lin1 = nn.Linear(NUM_PC_Train, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.lin3 = nn.Linear(256, NUM_CLS)

    def forward(self, x):
        pose, SH_, normal = x
        if self.use_feature:
            if self.useFPFH:
                fpfh0 = computeFPFH(pose.cpu(), normal.cpu()).to(pose.device)
                feature0 = self.rri(fpfh0, SH_)
            else:
                bfmap = BatchFeatureMAP(pose.clone().detach(), self.K, False)
                bfmap.batchNormal = normal
                index = torch.arange(pose.shape[1])
                simplifiedPointFeature = bfmap.computeBatchSPF(index)
                pose_dis = distance(pose)
                if self.use_GAT:
                    feature0 = self.rri(simplifiedPointFeature, torch.cat([SH_, pose_dis], dim=2),
                                        edge_index=bfmap.knn_indices)
                else:
                    feature0 = self.rri(simplifiedPointFeature, torch.cat([SH_, pose_dis], dim=2))
                # feature0 = self.rri(simplifiedPointFeature, SH_)
        else:
            feature0 = None
        pose0 = pose.clone().detach()
        normal0 = normal.clone().detach()
        sh0 = SH_.clone().detach()

        if self.useFPFH:
            pose1, feature1, normal1, sh1, fpfh1 = self.sa1_module(pose0, feature0, normal0, sh0, fpfh0)
            pose2, feature2, normal2, sh2, fpfh2 = self.sa2_module(pose1, feature1, normal1, sh1, fpfh1)
        else:
            pose1, feature1, normal1, sh1, fpfh1 = self.sa1_module(pose0, feature0, normal0, sh0)
            pose2, feature2, normal2, sh2, fpfh2 = self.sa2_module(pose1, feature1, normal1, sh1)
        if self.use_global_rri:
            if self.use_all_spf:
                bfmap2 = BatchFeatureMAP(pose2.clone().detach(), self.K, False)
                bfmap2.batchNormal = normal2
                index2 = torch.arange(pose2.shape[1])
                spf2 = bfmap2.computeBatchSPF(index2)
            else:
                pose_cluster = pose2.unsqueeze(dim=1)
                normal_cluster = normal2.unsqueeze(dim=1)
                point_feature = BatchFeatureMAP.computePF(pose_cluster, normal_cluster)  # B*N'*K*(c)
                spf2 = point_feature.squeeze(dim=1)
            pose_dis = distance(pose2)
            if self.use_GAT_GRRI:
                index = torch.arange(pose2.shape[1], device=pose2.device)
                index = index.unsqueeze(dim=0).repeat(pose2.shape[1], 1)
                index_2 = index[~torch.eye(index.shape[0], dtype=bool)].reshape(index.shape[0], -1)
                edge_index = index_2.unsqueeze(dim=0).repeat(pose2.shape[0], 1, 1)
                spf2_rri = self.global_rri(spf2, torch.cat([sh2, pose_dis], dim=2), edge_index=edge_index)
            else:
                spf2_rri = self.global_rri(spf2, torch.cat([sh2, pose_dis], dim=2))
            # spf2_rri = self.global_rri(spf2, sh2)
            # spf2_rri = self.global_rri(spf2, pose_dis)
            x = self.globalSa_module(torch.cat([feature2, spf2_rri], dim=2))
        else:
            x = self.globalSa_module(feature1)
        x = self.drop1(F.relu(self.bn1(self.lin1(x))))
        x = self.drop2(F.relu(self.bn2(self.lin2(x))))
        # x = F.relu(self.bn1(self.lin1(x)))
        # x = F.relu(self.bn1(self.lin1(x)))
        x = self.lin3(x)
        x = F.log_softmax(x, -1)
        return x
