import open3d as o3d
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from basic.EstimateNormals import BatchEstimateNormals
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from basic.common import unitSpher, zeroCenter, scalingPoints, singleZeroCenter, singleScalingPoint
import argparse
import torch_geometric.nn
from scipy.spatial.transform import Rotation as R
import multiprocessing
from basic.constant import *

torch.multiprocessing.set_sharing_strategy('file_system')
from basic.SphericalHarmonics import SphHarm


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


# def angle_axis(angle, axis):
#     r = R.from_rotvec(angle * axis)
#     return torch.from_numpy(r.as_dcm())


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudAzimuthalRotations(object):
    def __init__(self, axis=np.array([0.0, 0.0, 1.0])):
        self.axis = axis

    def get_rot(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)
        return rotation_matrix

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
            return points


class PointcloudArbitraryRotation(object):
    def __init__(self):
        pass

    def _get_angles(self):
        # rotation_angle_x = np.random.uniform() * 2 * np.pi
        # rotation_angle_y = np.random.uniform() * 2 * np.pi
        # rotation_angle_z = np.random.uniform() * 2 * np.pi
        angles = np.random.uniform(size=3) * 2 * np.pi
        return angles

    def get_rot(self):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        return rotation_matrix

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.002, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.std, self.std)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


def pca_preprocessing(points, type="train"):
    pca = torch.pca_lowrank(points)
    pcd_coordiate = pca[2]
    # x = torch.bmm(points, pcd_coordiate)
    # x = pcd_coordiate[:, :, 0]
    # y = pcd_coordiate[:, :, 1]
    # z = pcd_coordiate[:, :, 2]
    # z_ = torch.cross(x, y)
    # dist = torch.norm(z - z_, dim=1, keepdim=True)
    # fz = torch.where(dist > 1.0, -z, z)
    # fz = z_
    # pcd_coordiante_new = torch.empty_like(pcd_coordiate)
    # pcd_coordiante_new[:, :, 0] = x
    # pcd_coordiante_new[:, :, 1] = y
    # pcd_coordiante_new[:, :, 2] = fz
    # pcd_coordiate[:, :, 2] = fz
    if type == "train":
        new_coordinate = getCoordinate(pcd_coordiate)
        new_points = []
        for coordinate in new_coordinate:
            new_points.append(torch.bmm(points, coordinate))
    else:
        new_points = torch.bmm(points, pcd_coordiate)
    return new_points


def getCoordinate(coord):
    coordList = []
    coord0 = coord.clone().detach()
    coord1 = coord.clone().detach()
    coord1[:, :, 0] = coord[:, :, 0] * -1
    coord1[:, :, 1] = coord[:, :, 1] * -1
    coord1[:, :, 2] = coord[:, :, 2] * -1
    if False:
        coord2 = coord.clone().detach()
        coord2[:, :, 0] = coord[:, :, 0] * -1
        coord2[:, :, 1] = coord[:, :, 1] * 1
        coord2[:, :, 2] = coord[:, :, 2] * 1

        coord3 = coord.clone().detach()
        coord3[:, :, 0] = coord[:, :, 0] * 1
        coord3[:, :, 1] = coord[:, :, 1] * -1
        coord3[:, :, 2] = coord[:, :, 2] * 1

        coord4 = coord.clone().detach()
        coord4[:, :, 0] = coord[:, :, 0] * 1
        coord4[:, :, 1] = coord[:, :, 1] * 1
        coord4[:, :, 2] = coord[:, :, 2] * -1
        #
        coord5 = coord.clone().detach()
        coord5[:, :, 0] = coord[:, :, 0] * -1
        coord5[:, :, 1] = coord[:, :, 1] * -1
        coord5[:, :, 2] = coord[:, :, 2] * 1

        coord6 = coord.clone().detach()
        coord6[:, :, 0] = coord[:, :, 0] * -1
        coord6[:, :, 1] = coord[:, :, 1] * 1
        coord6[:, :, 2] = coord[:, :, 2] * -1

        coord7 = coord.clone().detach()
        coord7[:, :, 0] = coord[:, :, 0] * 1
        coord7[:, :, 1] = coord[:, :, 1] * -1
        coord7[:, :, 2] = coord[:, :, 2] * -1
    coordList.append(coord0)
    coordList.append(coord1)
    if False:
        coordList.append(coord2)
        coordList.append(coord3)
        coordList.append(coord4)
        coordList.append(coord5)
        coordList.append(coord6)
        coordList.append(coord7)
    return coordList


def reconstruction_and_sample(pcd):
    alpha = 0.03
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pcd)
    new_pcd.estimate_normals()
    radii = [0.05, 0.1, 0.2, 0.4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(new_pcd, 0.5)
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=new_pcd,
    #                                                                       radii=o3d.utility.DoubleVector(radii))
    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=new_pcd, depth=10)
    # inial_factor = int(NUM_PC_Train / NUM_PC_Test)
    # if inial_factor <= 5:
    #     inial_factor = 5
    #pcd2 = mesh.sample_points_poisson_disk(number_of_points=NUM_PC_Train, init_factor=5)
    pcd2 = mesh.sample_points_uniformly(number_of_points=NUM_PC_Train)
    return np.asarray(pcd2.points)


class ModelNet(Dataset):
    def __init__(self, num_points, partition='train', transforms=None, transformationType="Z", pertubation=False):
        self.data, self.label = ModelNet.load_data(partition)
        if partition == "train":
            self.num_points = NUM_PC_Train
        elif partition == "test":
            self.num_points = NUM_PC_Test
        else:
            raise Exception("partion is error")

        self.partition = partition
        self.transforms = transforms
        self.use_fps = True
        self.use_pertubation = pertubation
        self.use_partial = False
        self.MAX_Points = 2048
        self.delete_ratio = 0.1
        self.delete_Points = int(self.delete_ratio * self.MAX_Points)
        if transformationType == "Z":
            self.rot = PointcloudAzimuthalRotations()
        if transformationType == "S":
            self.rot = PointcloudArbitraryRotation()
        if self.use_pertubation:
            self.pertubation = PointcloudJitter()
        self.totensor = PointcloudToTensor()

    def get_reduced_data(self, item):
        if self.use_fps is False:
            pt_idxs = np.arange(0, self.data[item].shape[0])
            np.random.shuffle(pt_idxs)
            pointcloud = self.data[item][pt_idxs[:self.num_points]].copy()
        else:
            pointCloudPose = torch.from_numpy(self.data[item])
            pose_batch = pointCloudPose
            batch_index = torch.arange(1)
            batch_fps = batch_index.repeat(self.data[item].shape[0], 1).transpose(0, 1).reshape(-1)
            fps_idx = torch_geometric.nn.fps(pose_batch, batch_fps, self.num_points / self.MAX_Points)
            pointcloud = pose_batch[fps_idx].reshape(-1, 3).numpy()
        return pointcloud

    def __getitem__(self, item):
        if self.partition == "train":
            pointcloud = self.get_reduced_data(item)
        else:
            if self.use_partial:
                pt_idxs = np.arange(0, self.data[item].shape[0])
                np.random.shuffle(pt_idxs)
                select_data = self.data[item][pt_idxs[0]]
                dist = torch.norm(torch.from_numpy(select_data) - torch.from_numpy(self.data[item]), dim=1, p=2)
                knn = dist.topk(self.delete_Points, largest=False)
                index = np.arange(0, self.MAX_Points)
                reduces_index = np.delete(index, knn.indices.numpy())
                reduces_data = self.data[item][reduces_index]
                pointCloudPose = torch.from_numpy(reduces_data)
                pose_batch = pointCloudPose
                batch_index = torch.arange(1)
                batch_fps = batch_index.repeat(pointCloudPose.shape[0], 1).transpose(0, 1).reshape(-1)
                fps_idx = torch_geometric.nn.fps(pose_batch, batch_fps,
                                                 self.num_points / (self.MAX_Points - self.delete_Points))
                pointcloud = pose_batch[fps_idx].reshape(-1, 3).numpy()
            else:
                pointcloud = self.get_reduced_data(item)
                #if self.num_points < 1024:
                #    pointcloud = reconstruction_and_sample(pointcloud)

        label = self.label[item]
        # if self.transforms is not None:
        # pointcloud[:, 0] = pointcloud[:, 0] - np.mean(pointcloud[:, 0])
        # pointcloud[:, 1] = pointcloud[:, 1] - np.mean(pointcloud[:, 1])
        # pointcloud[:, 2] = pointcloud[:, 2] - np.mean(pointcloud[:, 2])
        pointcloud = self.totensor(pointcloud)
        # pointcloud = singleZeroCenter(pointcloud)
        # pointcloud = singleScalingPoint(pointcloud)
        pointcloud = self.rot(pointcloud)
        # if self.use_pertubation:
        #     pointcloud = self.pertubation(pointcloud)
        return pointcloud, label
        # if self.partition == "train":
        #     pointcloud = totensor(pointcloud)
        #     arzirot = PointcloudAzimuthalRotations()
        #     pointcloud = arzirot(pointcloud)
        #     return pointcloud, label
        # elif self.partition == "test":
        #     pointcloud = totensor(pointcloud)
        #     arbitaryRt = PointcloudArbitraryRotation()
        #     pointcloud = arbitaryRt(pointcloud)
        #     return pointcloud, label
        # else:
        #     raise ValueError("no corresponding partiation found")
        # else:
        #     return pointcloud, label
        # pointcloud = self.transforms(pointcloud)

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def load_data(partition):
        ModelNet.downloadModel40()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, '../data')
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    @staticmethod
    def downloadModel40():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, '../data')
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = os.path.basename(www)
            os.system('wget %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % zipfile)


class CreatModelNet40toSphericalHarmonics:

    def __init__(self, maxDegree, num_points=1024, num_workers=8, batch_size=8, partition='train', transforms=None,
                 transformationType="Z", pertubation=False, datafolder="data"):
        if partition == "train":
            num_points = NUM_PC_Train
        elif partition == "test":
            num_points = NUM_PC_Test
        else:
            raise Exception("Partian is not found")
        self.transformationType = transformationType
        self.maxDegree = maxDegree
        self.num_points = num_points
        if partition == "train":
            dataSets = DataLoader(ModelNet(partition='train', num_points=num_points, transforms=transforms,
                                           transformationType=transformationType,
                                           pertubation=pertubation),
                                  num_workers=num_workers,
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            partition = "test"
            dataSets = DataLoader(ModelNet(partition='test', num_points=num_points, transforms=transforms,
                                           transformationType=transformationType,
                                           pertubation=pertubation),
                                  num_workers=num_workers,
                                  batch_size=batch_size, shuffle=True, drop_last=False)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_folder = "../" + datafolder
        DATA_DIR = os.path.join(BASE_DIR, data_folder)
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        all_rif_data = []
        # all_shcoeff = []
        all_label = []
        all_points = []
        all_normals = []
        index = 0
        for i, dataset in enumerate(dataSets, 0):
            if i % 100 == 0:
                print(i)
            datas, labels = dataset

            # datas = unitSpher(datas) --> it can not work
            datas = zeroCenter(datas)
            # datas = scalingPoints(datas)
            if partition == "train":
                new_datas = pca_preprocessing(datas, type="train")
                for datas_ in new_datas:
                    rif, shCoeff = SphHarm(maxDegree, datas_).computeRotationInvarianceFeature(enableSVD=False)
                    normals = BatchEstimateNormals(datas_, 20, False).normal
                    all_label.append(labels)
                    # all_shcoeff.append(shCoeff)
                    all_rif_data.append(rif)
                    all_points.append(datas_)
                    all_normals.append(normals)
            else:
                datas = pca_preprocessing(datas, type="test")
                rif, shCoeff = SphHarm(maxDegree, datas).computeRotationInvarianceFeature(enableSVD=False)
                normals = BatchEstimateNormals(datas, 20, False).normal
                all_label.append(labels)
                # all_shcoeff.append(shCoeff)
                all_rif_data.append(rif)
                all_points.append(datas)
                all_normals.append(normals)

            if i % 100 == 0 and i is not 0:
                print("saving .... i=: ", i, "index: ", index)
                self.__save_h5(DATA_DIR, partition, index, all_rif_data, all_label, all_points, all_normals)
                all_rif_data = []
                # all_shcoeff = []
                all_label = []
                all_points = []
                all_normals = []
                index = index + 1
                print("finishing")
        if all_label is not None:
            print("saving .... at last 100, index:  ", index)
            self.__save_h5(DATA_DIR, partition, index, all_rif_data, all_label, all_points, all_normals)
            all_rif_data = []
            all_label = []
            all_points = []
            all_normals = []
            index = index + 1
            print("finishing")

    def __save_h5(self, DATA_DIR, partition, index, data, label, points, normals):
        DATA_DIR_SH = os.path.join(DATA_DIR, 'modelnet40_sh_h5')
        if not os.path.exists(DATA_DIR_SH):
            os.mkdir(DATA_DIR_SH)
        fileName = "/sh_" + partition + "_degree_" + str(self.maxDegree) + "_num_" + str(
            self.num_points) + "_" + self.transformationType + "_" + str(index) + ".h5"
        hf_train = h5py.File(DATA_DIR_SH + fileName, 'w')
        all_data = torch.cat(data, dim=0)
        all_label = torch.cat(label, dim=0)
        all_points = torch.cat(points, dim=0)
        all_normals = torch.cat(normals, dim=0)
        hf_train.create_dataset('sphericalHarmonics', data=all_data, dtype=float)
        hf_train.create_dataset('Labels', data=all_label, dtype=int)
        hf_train.create_dataset('points', data=all_points, dtype=float)
        hf_train.create_dataset('normals', data=all_normals, dtype=float)
        hf_train.close()


class ModelNet40SphericalHarmonics(Dataset):
    def __init__(self, maxDegree, num_points=1024, partition='train', transformationType="Z", dataSource="data"):
        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # self.dataSource = dataSource
        # filename = "../" + dataSource + "/modelnet40_sh_h5_1024/sh_"
        # self.baseFilePath = os.path.join(BASE_DIR, filename)
        if partition == "train":
            num_points = NUM_PC_Train
        elif partition == "test":
            num_points = NUM_PC_Test
        else:
            raise Exception("Partian is not found")
        self.dataSource = dataSource
        self.baseFileEnding = "sh_%s_degree_" + str(maxDegree) + "_num_" + str(
            num_points) + "_" + transformationType + "*" + ".h5"
        # self.file = baseFilePath + partition + baseFileEnding
        #
        # hf = h5py.File(self.file, 'r')
        # self.sh = hf.get("sphericalHarmonics")[:].astype('float32')
        # # self.shcoeff = hf.get("sphericalHarmonicsCoeff")[:].astype('float32')
        # self.labels = hf.get("Labels")[:].astype('int64')
        # self.points = hf.get("points")[:].astype('float32')
        # self.normals = hf.get("normals")[:].astype('float32')
        # self.partition = partition
        self.sh, self.labels, self.points, self.normals = self.load(partition)

    def __getitem__(self, item):
        shFeatures = self.sh[item]
        # shCoeff = self.shcoeff[item]
        point = self.points[item]
        normal = self.normals[item]
        label = self.labels[item]
        # if self.partition == 'train':
        #     np.random.shuffle(pointcloud)
        return shFeatures, point, normal, label

    def __len__(self):
        return self.sh.shape[0]

    def load(self, partition):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        datasourceStr = "../" + self.dataSource
        DATA_DIR = os.path.join(BASE_DIR, datasourceStr)
        local_sh = []
        local_label = []
        local_points = []
        local_normals = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_sh_h5', self.baseFileEnding % partition)):
            hf = h5py.File(h5_name, 'r')
            sh = hf.get("sphericalHarmonics")[:].astype('float32')
            labels = hf.get("Labels")[:].astype('int64')
            points = hf.get("points")[:].astype('float32')
            normals = hf.get("normals")[:].astype('float32')
            hf.close()
            local_sh.append(torch.from_numpy(sh))
            local_label.append(torch.from_numpy(labels))
            local_points.append(torch.from_numpy(points))
            local_normals.append(torch.from_numpy(normals))

        all_sh = torch.cat(local_sh, dim=0)
        all_labels = torch.cat(local_label, dim=0)
        all_points = torch.cat(local_points, dim=0)
        all_normals = torch.cat(local_normals, dim=0)
        return all_sh, all_labels, all_points, all_normals


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', transforms=None, transformationType="Z", pertubation=False):
        self.data, self.label = ModelNet.load_data(partition)
        if partition == "train":
            num_points = NUM_PC_Train
        elif partition == "test":
            num_points = NUM_PC_Test
        else:
            raise Exception("Partian is not found")
        self.num_points = num_points
        self.partition = partition
        self.transforms = transforms
        self.use_fps = False
        self.use_pertubation = pertubation
        self.use_partial = False
        self.MAX_Points = 2048
        self.delete_ratio = 0.1
        self.delete_Points = int(self.delete_ratio * self.MAX_Points)
        if transformationType == "Z":
            self.rot = PointcloudAzimuthalRotations()
        if transformationType == "S":
            self.rot = PointcloudArbitraryRotation()
        if self.use_pertubation:
            self.pertubation = PointcloudJitter()
        self.totensor = PointcloudToTensor()

    def get_reduced_data(self, item):
        if self.use_fps is False:
            pt_idxs = np.arange(0, self.data[item].shape[0])
            np.random.shuffle(pt_idxs)
            pointcloud = self.data[item][pt_idxs[:self.num_points]].copy()
        else:
            pointCloudPose = torch.from_numpy(self.data[item])
            pose_batch = pointCloudPose
            batch_index = torch.arange(1)
            batch_fps = batch_index.repeat(self.data[item].shape[0], 1).transpose(0, 1).reshape(-1)
            fps_idx = torch_geometric.nn.fps(pose_batch, batch_fps, NUM_PC / self.MAX_Points)
            pointcloud = pose_batch[fps_idx].reshape(-1, 3).numpy()
        return pointcloud

    def __getitem__(self, item):
        pc = self.get_reduced_data(item)
        pc = self.totensor(pc)
        pc = self.rot(pc)
        if self.num_points < 1024:
            pc = torch.from_numpy(reconstruction_and_sample(pc)).float()
        return pc, self.label[item]

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Point Cloud classification DataSet preprocessing')
    parser.add_argument('--maxdegree', type=int, default='20', help="maxmum degree ")
    parser.add_argument("--pertubation", type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='use_pertubation')
    parser.add_argument("--train", nargs="+", default=["Z", "S"])
    parser.add_argument("--test", nargs="+", default=["Z", "S"])
    parser.add_argument("--datafolder", type=str, default="data", help="give the datafolder name")
    args = parser.parse_args()
    if args.train[0] == "Z":
        print("train in Z")
        model_train_Z = CreatModelNet40toSphericalHarmonics(int(args.maxdegree), partition="train",
                                                            transforms=None,
                                                            transformationType="Z",
                                                            pertubation=args.pertubation,
                                                            datafolder=args.datafolder)

    if args.train[1] == "S":
        print("train in S")
        model_train_S = CreatModelNet40toSphericalHarmonics(int(args.maxdegree), partition="train",
                                                            transforms=None,
                                                            transformationType="S",
                                                            pertubation=args.pertubation,
                                                            datafolder=args.datafolder)
    if args.test[0] == "Z":
        print("test in Z")
        model_test_Z = CreatModelNet40toSphericalHarmonics(int(args.maxdegree), partition="test",
                                                            transforms=None,
                                                            transformationType="Z",
                                                            pertubation=args.pertubation,
                                                            datafolder=args.datafolder)
    if args.test[1] == "S":
        print("test in S")
        model_test_S = CreatModelNet40toSphericalHarmonics(int(args.maxdegree), partition="test",
                                                            transforms=None,
                                                            transformationType="S",
                                                            pertubation=args.pertubation,
                                                            datafolder=args.datafolder)

    
