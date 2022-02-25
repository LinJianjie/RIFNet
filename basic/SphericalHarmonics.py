import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import torch
from scipy import special as sp


class SphHarm:
    def __init__(self, maxDegree, points):  # points=[batch_size, npoint, 3]
        self.points = points
        self.batchSize = self.points.shape[0]
        self.nPoints = self.points.shape[1]
        self.dim = self.points.shape[2]
        self.maxSHDegree = maxDegree
        self.shFeature = (self.maxSHDegree + 1) ** 2

    @staticmethod
    def getNumOfSH(maxDegree):
        return maxDegree + 1

    def unitSpher(self):
        norm = torch.norm(self.points, dim=2, p=2, keepdim=True)
        self.points = self.points / norm

    def zeroCenter(self):
        mean = self.points.mean(1).reshape(self.batchSize, 1, self.dim)
        self.points = self.points - mean

    def cart2sph(self):
        # self.zeroCenter()
        # self.unitSpher()
        xsqPlusYsq = self.points[:, :, 0] ** 2 + self.points[:, :, 1] ** 2
        r = torch.sqrt(xsqPlusYsq + self.points[:, :, 2] ** 2)
        theta = torch.atan2(self.points[:, :, 1], self.points[:, :, 0])  # [0, 2*pi]--> azimuthal
        phi = torch.atan2(torch.sqrt(xsqPlusYsq), self.points[:, :, 2])  # [0, pi] -->inclination coordiantes
        return r, theta, phi

    def getShRealHarm(self, m, n, theta_, phi_):
        theta = theta_.reshape(-1, 1)  # [batch*N,1 ]
        phi = phi_.reshape(-1, 1)  # [batch*N, 1]
        ynm = sp.sph_harm(m, n, theta, phi)  #
        ynm_ = np.where(m == 0, ynm, np.sqrt(2) * (-1) ** m * ynm)
        # print("pose: ", self.points[0, 0:2, :])
        # print("result:", ynm_[0:2, 0:4])
        # if m != 0:
        #     ynm_plus = np.sqrt(2) * (-1) ** m * np.real(ynm)  # [batch*N, 1]
        #     ynm_neg = np.sqrt(2) * (-1) ** m * np.imag(ynm)  # [batch*N, 1]
        # else:
        #     ynm_plus = ynm
        #     ynm_neg = None  # [batch*N, 1]
        ynm_plus = np.real(ynm_)
        ynm_neg = np.imag(ynm_)
        coeff = []
        curinter = 0
        for i in range(self.maxSHDegree + 1):
            coeff.append(ynm_neg[:, np.arange(curinter + 1, curinter + i + 1)])
            coeff.append(ynm_plus[:, np.arange(curinter, curinter + i + 1)])
            curinter += i + 1
        coeff = np.concatenate(coeff, axis=1)
        coeff = coeff.reshape(self.batchSize, self.nPoints, self.shFeature)
        return coeff

    @staticmethod
    def monteCarloIntergration(r, harmCoeff):
        return (r * harmCoeff).mean(1) * (4 * np.pi)

    def computeSHCoefficent(self):
        r, theta, phi = self.cart2sph()
        r = r.reshape(self.batchSize, self.nPoints, 1)
        theta_numpy = theta.data.numpy()
        phi_numpy = phi.data.numpy()
        # for n in range(self.maxSHDegree + 1):
        #     for m in range(n + 1):
        #         ynm_plus, ynm_neg = SH.getShRealHarm(m, n, theta_numpy, phi_numpy)  # [batch*N, 1]
        #         ynm_plus = ynm_plus.reshape(self.batchSize, self.nPoints, 1)
        #         A.append(ynm_plus)
        #         if m > 0:
        #             ynm_neg = ynm_neg.reshape(self.batchSize, self.nPoints, 1)
        #             A.append(ynm_neg)
        order = np.concatenate([np.arange(n + 1) for n in range(self.maxSHDegree + 1)])
        degree = np.concatenate([np.full(n + 1, n) for n in range(self.maxSHDegree + 1)])
        sharmCoeff = torch.from_numpy(self.getShRealHarm(order, degree, theta_numpy, phi_numpy)).float()
        sharmfactor = SphHarm.monteCarloIntergration(r, sharmCoeff).unsqueeze(dim=1).repeat(1, sharmCoeff.shape[1], 1)
        return sharmfactor, sharmCoeff

    def computeRotationInvarianceFeature(self, enableSVD=True):
        sharmFactor, sharmCoeff = self.computeSHCoefficent()
        usesph1 = False
        if usesph1:
            reconstruction = sharmFactor * sharmFactor
        else:
            reconstruction = sharmFactor * sharmCoeff
        #
        # reconstruction = sharmCoeff
        currIter = 0
        rif = torch.zeros([self.batchSize, self.nPoints, self.maxSHDegree + 1])
        for i in range(self.maxSHDegree + 1):
            temp = reconstruction[:, :, range(currIter, currIter + 2 * i + 1)]
            if usesph1:
                rif[:, :, i] = torch.sum(temp, dim=2, keepdim=False)
            else:
                # # rif[:, :, i] = torch.norm(temp, p=2, dim=2, keepdim=False)
                rif[:, :, i] = torch.norm(temp, p=2, dim=2, keepdim=False) * torch.norm(self.points, p=2, dim=2,
                                                                                        keepdim=False)
                # rif[:, :, i] = torch.norm(temp, p=2, dim=2, keepdim=False)
                # sum_temp = torch.sum(temp, dim=2, keepdim=True)
                # rif[:, :, i] = torch.norm(sum_temp, p=2, dim=2, keepdim=False) * torch.norm(self.points, p=2, dim=2,
                #                                                                             keepdim=False)
                # rif[:, :, i] = torch.sum(temp ** 2, dim=2, keepdim=False) * torch.norm(self.points, p=2, dim=2,
                #                                                                       keepdim=False)
                # rif[:, :, i] = torch.norm(temp, p=2, dim=2, keepdim=False)
                # rif[:, :, i] = torch.sum(temp**2, dim=2, keepdim=False)

            currIter += 2 * i + 1
            if enableSVD:
                u, _, v = torch.svd(rif)
                rif = torch.bmm(u, v)
        return rif, sharmCoeff


if __name__ == "__main__":
    pass
