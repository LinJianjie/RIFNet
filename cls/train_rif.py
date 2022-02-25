import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import argparse
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from basic.sphspFModdel import SPHSPFModel
from basic.PointNet_DGCNN import PointNet, DGCNN
from basic.createDataset import ModelNet40SphericalHarmonics, ModelNet40
import yaml
import config
from basic.pytorch_utils import *
from basic.constant import *


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud classification')
    parser.add_argument('--netConfig', type=str, metavar='NetConfig_config', default="RIFCofig.yaml",
                        help='the net work path of robot: eg. RIFConfig.yaml')
    parser.add_argument('--dataset_type', type=str, default='sphericalModel40',
                        help="dataset type sphericalModel40|modelnet40")
    parser.add_argument('--datamaxdegree', type=int, default='20', help="maxmum degree ")
    parser.add_argument('--maxdegree', type=int, default='20', help="maxmum degree ")
    parser.add_argument('--best_model_name', type=str, default='best_model_spf_sph_20', help="the name of checkpoints")
    parser.add_argument('--tag', type=str, help="show the git tag")
    parser.add_argument("--trainRot", type=str, default="Z", help="which rotation should be used for train")
    parser.add_argument("--testRot", type=str, default="Z", help="which rotation should be used for testing")
    parser.add_argument("--cuda", type=str, default="0", help="which cuda used")
    parser.add_argument("--BatchSize", type=int, default=8, help="the batchsize")
    parser.add_argument('--load_checkPoints', type=lambda x: not (str(x).lower() == 'false'), default=False,
                        help='load_checkpoints')
    parser.add_argument('--checkPointsModel', type=str, help='load_checkpoints model')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of episode to train ')
    parser.add_argument('--dataSource', type=str, default="data", help="dataSource")
    parser.add_argument('--logfilename', type=str, default="log", help="create_a_logfilename")
    parser.add_argument('--model', type=str, default='rifNet', metavar='N',
                        choices=['rifNet', 'pointnet', 'dgcnn'],
                        help='Model to use, [rifNet, pointnet, dgcnn]')
    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    with open(os.path.join(config.getDataConfigPath(), args_.netConfig)) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    has_cuda = not args['NetWorkSetting']['has_cuda'] and torch.cuda.is_available()
    # torch.manual_seed(args['NetWorkSetting']['seed'])
    if args_.dataset_type == 'sphericalModel40':
        print("DataSet -----> ModelNet40SphericalHarmonics")
        train_loader = DataLoader(
            ModelNet40SphericalHarmonics(maxDegree=int(args_.datamaxdegree),
                                         num_points=NUM_PC,
                                         partition='train',
                                         transformationType=args_.trainRot,
                                         dataSource=args_.dataSource),
            num_workers=args['NetWorkSetting']['dataLoader']['num_workers'],
            batch_size=args_.BatchSize,
            shuffle=args['NetWorkSetting']['train']['shuffle'],
            drop_last=args['NetWorkSetting']['train']['drop_last'])
        test_loader = DataLoader(
            ModelNet40SphericalHarmonics(maxDegree=int(args_.datamaxdegree),
                                         num_points=NUM_PC,
                                         partition='test',
                                         transformationType=args_.testRot,
                                         dataSource=args_.dataSource),
            num_workers=args['NetWorkSetting']['dataLoader']['num_workers'],
            batch_size=args_.BatchSize,
            shuffle=args['NetWorkSetting']['test']['shuffle'],
            drop_last=args['NetWorkSetting']['test']['drop_last'])
    elif args_.dataset_type == 'modelnet40':
        print("DataSet ------>ModelNet40")
        train_loader = DataLoader(
            ModelNet40(num_points=NUM_PC, partition='train', transformationType=args_.trainRot),
            num_workers=args['NetWorkSetting']['dataLoader']['num_workers'],
            batch_size=args_.BatchSize,
            shuffle=args['NetWorkSetting']['train']['shuffle'],
            drop_last=args['NetWorkSetting']['train']['drop_last'])
        test_loader = DataLoader(
            ModelNet40(num_points=NUM_PC, partition='test', transformationType=args_.testRot),
            num_workers=args['NetWorkSetting']['dataLoader']['num_workers'],
            batch_size=args_.BatchSize,
            shuffle=args['NetWorkSetting']['test']['shuffle'],
            drop_last=args['NetWorkSetting']['test']['drop_last'])
    else:
        raise Exception("no Dataset is found")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args_.model == "rifNet":
        print("RIF Net")
        model = SPHSPFModel(int(args_.maxdegree))
    elif args_.model == "pointnet":
        print("PointNet")
        model = PointNet()
    elif args_.model == 'dgcnn':
        print("DGCNN")
        model = DGCNN()
    else:
        raise Exception("Not implemented")
    if torch.cuda.device_count() > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        # cuda_index = "cuda:" + args_.cuda
        # device = torch.device(cuda_index)
        model.to(device)

    if not os.path.exists(os.path.join(os.getcwd(), "../checkpoints")):
        os.makedirs(os.path.join(os.getcwd(), "../checkpoints"))

    if args['NetWorkSetting']['train']['use_sgd']:
        print("Optimizer ------> SGD")
        opt = optim.SGD(model.parameters(), lr=args['NetWorkSetting']['train']['lr'] * 100,
                        momentum=args['NetWorkSetting']['train']['momentum'])
    else:
        print("Optimizer ------> Adam")
        opt = optim.Adam(model.parameters(), lr=args['NetWorkSetting']['train']['lr'],
                         betas=(0.9, 0.999)
                         )

    if args_.checkPointsModel is not None:
        load_checkpoint(model, opt,
                        os.path.join(os.getcwd(), "../checkpoints/" + args_.checkPointsModel + ".pth"))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args['NetWorkSetting']['train']['epochs'],
                                                           eta_min=args['NetWorkSetting']['train']['lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.7)
    trainer = TorchTrainer(model=model, loss_function=F.nll_loss,
                           optimizer=opt,
                           checkpoint_name="rif",
                           lr_scheduler=scheduler,
                           maxdegree=int(args_.maxdegree),
                           loggerfilename=args_.dataSource + "/" + args_.logfilename + "_" + args_.tag + ".log")
    trainer.train(start_epoch=args['NetWorkSetting']['train']['start_iter'],
                  n_epochs=args_.epochs,
                  device=device,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  batch_size=args_.BatchSize,
                  best_model_name=args_.dataSource + "/" + args_.best_model_name + "_" + args_.tag + ".pth",
                  tag=args_.tag)

