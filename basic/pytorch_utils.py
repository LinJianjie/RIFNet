import os
import shutil
import logging
import torch
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from basic.logger import Logger
import time


def save_checkpoint(state, is_best, filename="checkpoint", bestname="model_best"):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    # filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return None
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def cal_loss(pred, label, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = label.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # creat a one-hot vector
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=-1)

        # loss = -(one_hot * log_prb).sum(dim=1).mean()
        loss = F.nll_loss(log_prb, label)
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class TorchTrainer:
    def __init__(self, model=None, loss_function=None, optimizer=None, checkpoint_name="ckpt", best_name="best",
                 lr_scheduler=None,
                 eval_frequency=-1,
                 maxdegree=20,
                 loggerfilename="log.log"):
        self.model = model
        self.loss_function = cal_loss
        self.optimizer = optimizer
        self.checkpoint_name = checkpoint_name
        self.best_name = best_name
        self.lr_scheduler = lr_scheduler
        self.maxdegree = maxdegree
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model can not be none")
        if self.loss_function is None:
            raise ValueError("loss_function can not be none")
        if self.optimizer is None:
            raise ValueError("optimizer can not be none")
        self.global_epoch = 0
        self.global_step = 0
        self.best_instance_acc = 0.0
        self.best_class_acc = 0.0
        self.mean_correct = []
        self.best_epoch = 0
        self.n_epochs = 0
        self.Logger = Logger(filename=loggerfilename)

    def __train_it(self, train_loader, test_loader, device, at_epoch, n_epochs, batch_size):
        train_correct = 0
        count = 0.0
        train_loss = 0
        train_pred = []
        train_true = []

        self.model.train()
        for i, dataset in enumerate(train_loader):
            if 2 == len(dataset):
                points, label = dataset
                shPoint = None
                normal = None
            else:
                shPoint, points, normal, label = dataset
            if shPoint is not None:
                shPoint = shPoint[:, :, :(self.maxdegree + 1)]
                shPoint = shPoint.to(device)
            if points is not None:
                points = points.to(device)
            if normal is not None:
                normal = normal.to(device)
            if label is not None:
                label = label.to(device).squeeze()
            # shPoint.cuda()
            # points.cuda()
            # normal.cuda()
            # label.cuda().squeeze()
            self.optimizer.zero_grad()
            self.model.train()
            input = (points, shPoint, normal)
            pred = self.model(input)
            loss = F.nll_loss(pred, label)
            # loss = self.loss_function(pred, label)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            pred_choice = pred.max(1)[1]
            train_correct += pred_choice.eq(label).sum().item()
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(pred_choice.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        # train_instance_acc = train_correct / len(train_loader)
        # train_instance_loss = train_loss / len(train_loader)
        # print('at epoch %i, the train_instance acc :%f, train_loss: %f' % (
        #    at_epoch, train_instance_acc, train_instance_loss))
        # print('Train %d/%d, loss: %.6f, train acc: %.6f, avg acc: %.6f' % (at_epoch, n_epochs,
        #                                                                    train_loss * 1.0 / count,
        #                                                                    metrics.accuracy_score(
        #                                                                        train_true,
        #                                                                        train_pred),
        #                                                                    metrics.balanced_accuracy_score(
        #                                                                        train_true,
        #                                                                        train_pred)))
        self.Logger.INFO('Train %d/%d, loss: %.6f, train acc: %.6f',
                         at_epoch,
                         n_epochs,
                         train_loss * 1.0 / count,
                         metrics.accuracy_score(train_true, train_pred))
        return loss

    def eval_epoch(self, data_loader, device, epoch, best_model_name="best_model.pth", tag="v1.0"):
        self.model.eval()
        correct = 0
        test_pred = []
        test_true = []
        count = 0.0
        test_loss = 0
        batch_size = 0
        for i, dataset in enumerate(data_loader):
            if 2 == len(dataset):
                points, label = dataset
                shPoint = None
                normal = None
            else:
                shPoint, points, normal, label = dataset
            if shPoint is not None:
                shPoint = shPoint[:, :, :(self.maxdegree + 1)]
                shPoint = shPoint.to(device)

            batch_size, N, D = points.shape

            points = points.to(device)
            if normal is not None:
                normal = normal.to(device)
            label = label.to(device).squeeze()
            # shPoint.cuda()
            # points.cuda()
            # normal.cuda()
            # label.cuda().squeeze()
            with torch.no_grad():
                input = (points, shPoint, normal)
                # t0 = time.time()
                pred = self.model(input)
                # print('{} seconds'.format((time.time() - t0)/shPoint.shape[0]))
                loss = F.nll_loss(pred, label)
                # loss = self.loss_function(pred, label)
                pred_choice = pred.max(1)[1]
                count += points.shape[0]
                test_loss += loss.item() * points.shape[0]
                test_true.append(label.cpu().numpy())
                test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        # print('Test %d/%d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best acc: %.6f' % (epoch, self.n_epochs,
        #                                                                                       test_loss * 1.0 / count,
        #                                                                                       test_acc,
        #                                                                                       avg_per_class_acc,
        #                                                                                       self.best_instance_acc))
        self.Logger.INFO('Test %d/%d, loss: %.6f, test acc: %.6f, best acc: %.6f',
                         epoch,
                         self.n_epochs,
                         test_loss * 1.0 / count,
                         test_acc,
                         self.best_instance_acc)

        if test_acc > self.best_instance_acc:
            self.best_instance_acc = test_acc
            self.best_epoch = epoch
            checkpointPath = os.path.join(os.getcwd(), '../checkpoints')
            if not os.path.exists(checkpointPath):
                os.mkdir(checkpointPath)
            # savepath = os.path.join(os.getcwd(), "../checkpoints/" + best_model_name)
            savepath = os.path.join(os.getcwd(), best_model_name)
            if self.best_instance_acc >= 0.87:
                state = {
                    "tag": tag,
                    "batch_size": batch_size,
                    "epoch": self.best_epoch,
                    "instance_acc": self.best_instance_acc,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict()}
                torch.save(state, savepath)

    def train(self, start_epoch, n_epochs, device, train_loader, test_loader=None, best_loss=0.0, batch_size=8,
              best_model_name="best_model.pth", tag="v1.0"):
        self.n_epochs = n_epochs
        for at_epoch in range(start_epoch, n_epochs):
            loss = self.__train_it(train_loader=train_loader, test_loader=test_loader, device=device, at_epoch=at_epoch,
                                   n_epochs=n_epochs,
                                   batch_size=batch_size)
            self.eval_epoch(test_loader, device, at_epoch, best_model_name, tag)
