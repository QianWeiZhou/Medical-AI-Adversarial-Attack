# Copyright 2021 QianWei Zhou
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''24
'''
import os
# from pycparser.c_ast import ParamList
# from numpy.dual import norm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # if use '2,1', then in pytorch, gpu2 has id 0
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL

import DiscriminatorVGG1728 as dis  # change
import util as ul
import ImageDataset4f1728 as ImD  # change
import ImageDataset5f1728 as ID  # change
import torch.cuda
import torch.utils.data
import focalloss
import torchvision.transforms as tf

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # @UndefinedVariable
_VF_ = dis.getVggFe()

if cuda:
    _VF_ = _VF_.cuda()

fl_mm = 0
fl_weight = np.array([1 / 2, 1 / 2])

print('fl_weight = {}'.format(fl_weight.tolist()))
fl = focalloss.FocalLoss(gamma=2, alpha=fl_weight.tolist())

preTrained = None
checkfolder = './checkfolder-Classifier/'
# ====if its not None, first model loading will try to read models from this folder===================
# ReadModelFrom = './../Breast2020Check/checkfolderClas-1728pal134-TT/'
ReadModelFrom = None
# ============otherwise, it will read from checkfolder.===============================================
DataFolder = './Samples/realImages/'
os.system('mkdir -p {}/'.format(checkfolder))

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
n_classes = 2
imH = 1728
imW = 1408
c_dim = 3
k = 10
p = 4
batch_size = 4 # should be 4
bestP = 0

pretransform = tf.Compose([
    tf.RandomVerticalFlip(),
    tf.RandomAffine(45, translate=(0, 0), scale=(0.5, 2), shear=45, resample=PIL.Image.BILINEAR, fillcolor=0),  # -D5
])



class FNnet(nn.Module):
    def __init__(self):
        super(FNnet, self).__init__()
        preNet = dis.DisNet1DpCov132(z_dim=2048, N=(8192, 2048, 256), C=1, GN=32)
        try:
            preNet.load_state_dict(torch.load(preTrained, map_location=torch.device('cpu')))
        except:
            print('===>no PreTrained data found, use random initialization!')
        self.linSq = preNet.lins
        self.Cov = preNet.Cov
        self.CovMax = preNet.CovMax

        self.linSq.__delitem__(10)

    def forward(self, x):
        x = self.Cov(x)
        sumBefore = torch.sum(x)  # @UndefinedVariable
        x = self.CovMax(x)
        xn = torch.numel(x)  # @UndefinedVariable
        sumAfter = torch.sum(x)  # @UndefinedVariable
        sl = (sumBefore - sumAfter) / xn

        x = x.view(-1, 2048)
        return self.linSq(x), sl


def load_data():
    print('loading data!')
    global trainset
    trainset = ID.ImageDataset(transform=pretransform, Posi=4, rootFolder=DataFolder, gn=0)
    validset = ImD.ImageDataset(transform=None, setType=1, rootFolder=DataFolder)
    testset = ImD.ImageDataset(transform=None, setType=2, rootFolder=DataFolder)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True, drop_last=True,
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size,
                                               shuffle=False, drop_last=False,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False, drop_last=False,
                                              **kwargs)

    return train_loader, valid_loader, test_loader

def train(modelList, paramList, data_loader, epoch=0):
    global fl
    global fl_mm
    global fl_weight
    global check
    global _step_
    if cuda:
        torch.cuda.set_device(0)

    for param_group in paramList['FN'].param_groups:
        print(param_group['lr'])

    for ml in modelList:
        ml.train()

    FN = modelList[0]
    CN = modelList[1]

    iter_count = 0.0
    NLL_loss_a = 0.0
    wLoss = 0.0
    sl = 0.0
    corrects = np.array([0, 0])
    totalPtarget = np.array([0, 0])

    for batch_index, batch_data in enumerate(data_loader, 0):
        iter_count += 1.0
        ul.printProgressBar(iter_count, data_loader.__len__());
        pX, nX, pT, nT = batch_data

        pX, nX, pT, nT = Variable(pX, requires_grad=True), Variable(nX, requires_grad=True), Variable(pT), Variable(nT)
        if cuda:
            pX, nX, pT, nT = pX.cuda(), nX.cuda(), pT.cuda(), nT.cuda()

        if _step_ == 1:
            paramList['CN'].zero_grad()

            vpX = _VF_(pX)
            vnX = _VF_(nX)
            pFeature, slp = FN(vpX)
            nFeature, sln = FN(vnX)
            slpn = torch.mean(slp + sln) / 2  # @UndefinedVariable
            poutput = CN(pFeature)
            noutput = CN(nFeature)
            ploss = fl(poutput, pT)
            nloss = fl(noutput, nT)

            real_grad_out = Variable(Tensor(pFeature.size(0), 2).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(poutput,
                                      pX,
                                      real_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(nFeature.size(0), 2).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(noutput,
                                      nX,
                                      fake_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2  # @UndefinedVariable

            loss = ploss + nloss + div_gp  

            loss.backward()
            paramList['CN'].step()
            wLoss += div_gp.data.cpu().numpy()
            sl += slpn.data.cpu().numpy()

        if _step_ == 2:
            paramList['FN'].zero_grad()
            paramList['CN'].zero_grad()

            vpX = _VF_(pX)
            vnX = _VF_(nX)

            pFeature, slp = FN(vpX)
            nFeature, sln = FN(vnX)

            slpn = torch.mean(slp + sln) / 2  # @UndefinedVariable
            poutput = CN(pFeature)
            noutput = CN(nFeature)

            ploss = fl(poutput, pT)
            nloss = fl(noutput, nT)

            real_grad_out = Variable(Tensor(vpX.size(0), 2).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(poutput,
                                      pX,
                                      real_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(vnX.size(0), 2).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(noutput,
                                      nX,
                                      fake_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2  # @UndefinedVariable

            loss = ploss + nloss + div_gp  # +slpn/10

            loss.backward()

            paramList['CN'].step()
            paramList['FN'].step()
            wLoss += div_gp.data.cpu().numpy()
            sl += slpn.data.cpu().numpy()

        if _step_ == 3:
            paramList['FN'].zero_grad()
            paramList['CN'].zero_grad()
            paramList['VN'].zero_grad()

            vpX = _VF_(pX)
            vnX = _VF_(nX)
            pFeature, slp = FN(vpX)
            nFeature, sln = FN(vnX)
            slpn = torch.mean(slp + sln) / 2  # @UndefinedVariable
            poutput = CN(pFeature)
            noutput = CN(nFeature)
            ploss = fl(poutput, pT)
            nloss = fl(noutput, nT)

            real_grad_out = Variable(Tensor(vpX.size(0), 2).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(poutput,
                                      pX,
                                      real_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(vnX.size(0), 2).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(noutput,
                                      nX,
                                      fake_grad_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2  # @UndefinedVariable

            loss = ploss + nloss + div_gp

            loss.backward()
            paramList['CN'].step()
            paramList['FN'].step()
            paramList['VN'].step()
            wLoss += div_gp.data.cpu().numpy()
            sl += slpn.data.cpu().numpy()

        ppred = poutput.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pcorrect = ppred.eq(pT.data.view_as(ppred)).cpu().numpy()
        ptnp = pT.data.view_as(ppred).cpu().numpy()

        npred = noutput.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        ncorrect = npred.eq(nT.data.view_as(ppred)).cpu().numpy()
        ntnp = nT.data.view_as(npred).cpu().numpy()

        correct = np.concatenate((pcorrect, ncorrect))
        tnp = np.concatenate((ptnp, ntnp))

        for i in range(len(corrects)):
            corrects[i] += np.sum(correct[tnp == i])
            totalPtarget[i] += np.double(np.sum(tnp == i))

        NLL_loss_a += loss.cpu().data.numpy()

    NLL_loss_a = NLL_loss_a / iter_count

    totalC = 1 - np.sum(corrects) / np.sum(totalPtarget)
    corrects = corrects / totalPtarget

    weight = np.array((1 - corrects) / np.sum(1 - corrects))
    fl_weight = fl_weight * (1 - fl_mm) + weight * fl_mm
    fl = focalloss.FocalLoss(gamma=2, alpha=fl_weight.tolist())

    wLoss = wLoss / iter_count
    sl = sl / iter_count

    return [corrects[0], corrects[1], totalC, NLL_loss_a, np.log10(wLoss + 1), sl]


def testAUC(modelList, data_loader, epoch=0):
    FN = modelList[0]
    CN = modelList[1]

    FN.eval()
    CN.eval()
    _VF_.eval()

    iter_count = 0
    TRs = np.zeros([2, 201])
    totalPtarget = np.array([0, 0])
    for batch_index, batch_data in enumerate(data_loader, 0):
        iter_count += 1.0
        ul.printProgressBar(iter_count, data_loader.__len__());

        cX, cT, cID = batch_data
        cX, cT = Variable(cX), Variable(cT)
        if cuda:
            cX, cT = cX.cuda(), cT.cuda()

        vcX = _VF_(cX)
        fFeature, slx = FN(vcX)
        output = CN(fFeature)
        output = F.softmax(output, dim=1)
        output = output.data.cpu()
        tnp = cT.data.view(output.size(0), 1).cpu().numpy()
        cT = cT.data.view(output.size(0), 1).cpu()
        for j in range(len(totalPtarget)):
            totalPtarget[j] += np.double(np.sum(tnp == j))
        for i in range(0, 201):
            tempOut = output.clone()
            tempOut[:, 1] = tempOut[:, 1] + i / 100.0 - 1
            pred = tempOut.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct = pred.eq(cT).numpy()

            for j in range(len(totalPtarget)):
                TRs[j][i] += np.sum(correct[tnp == j])

    for j in range(len(totalPtarget)):
        TRs[j] = TRs[j] / totalPtarget[j]

    TRs = np.flip(TRs, 1)
    TRs[1, :] = 1 - TRs[1, :]
    temp = np.zeros((1, TRs.shape[1]))
    temp[0, 1:] = TRs[1, :-1]
    temp[0, :] = TRs[1, :] - temp[0, :]
    AUC = np.sum(TRs[0, :] * temp[0, :])

    return AUC


def generate_model(train_loader, test_loader, vali_loader, modelState):
    global ReadModelFrom
    global trainset
    global bestP
    global _VF_

    e_a = modelState['epoch_Add']

    print('define models')

    FN = FNnet()

    CN = dis.DisNet(z_dim=256, N=None, dp=0, C=2, GN=0, pDp=0)

    if not isinstance(FN, nn.DataParallel):
        FN = nn.DataParallel(FN)
    if not isinstance(CN, nn.DataParallel):
        CN = nn.DataParallel(CN)
    if not isinstance(_VF_, nn.DataParallel):
        _VF_ = nn.DataParallel(_VF_)

    if modelState['OEGQDDL3']:
        print('loading models ....')
        if ReadModelFrom:
            print('reading model from {}'.format(ReadModelFrom))
        else:
            ReadModelFrom = checkfolder
        oegFolder = modelState['oegFolder']
        genNum = modelState['genNum']
        print(oegFolder.format(ReadModelFrom, 'FN', genNum))
        CN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom, 'CN', genNum), map_location=torch.device('cpu')))
        try:
            FN.load_state_dict(
                torch.load(oegFolder.format(ReadModelFrom, 'FN', genNum), map_location=torch.device('cpu')))
        except:
            print('====> no FN history found! use default initialization.')
        if _step_ == 3:
            try:
                _VF_.cpu()
                _VF_.load_state_dict(
                    torch.load(oegFolder.format(ReadModelFrom, 'VN', genNum), map_location=torch.device('cpu')))
            except:
                print('====> no VN history found! use default initialization.')

    print('move to GPU')

    if cuda:
        FN.cuda()
        CN.cuda()
        _VF_.cuda()

    FN_lr = modelState['FN_lr']
    CN_lr = modelState['CN_lr']

    FN_opt = optim.Adam(FN.parameters(), lr=FN_lr, betas=(0.5, 0.9))
    CN_opt = optim.Adam(CN.parameters(), lr=CN_lr, betas=(0.5, 0.9))
    VN_opt = optim.Adam(_VF_.parameters(), lr=FN_lr / 10, betas=(0.5, 0.9))

    if modelState['OEGQDDL3']:
        try:
            FN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom, 'FN', modelState['genNum']),
                                   map_location=torch.device('cpu'))
            FN_opt.load_state_dict(FN_opt_st)
        except:
            print('====> no FN_optim history found! use default initialization.')
        try:
            CN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom, 'CN', modelState['genNum']),
                                   map_location=torch.device('cpu'))
            CN_opt.load_state_dict(CN_opt_st)
        except:
            print('====> no CN_optim history found! use default initialization.')

        try:
            VN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom, 'VN', modelState['genNum']),
                                   map_location=torch.device('cpu'))
            VN_opt.load_state_dict(VN_opt_st)
        except:
            print('====> no VN_optim history found! use default initialization.')

        for param_group in FN_opt.param_groups:
            param_group['lr'] = FN_lr
            param_group['betas'] = (0.5, 0.9)
        for param_group in CN_opt.param_groups:
            param_group['lr'] = CN_lr
            param_group['betas'] = (0.5, 0.9)
        for param_group in VN_opt.param_groups:
            param_group['lr'] = FN_lr / 10
            param_group['betas'] = (0.5, 0.9)

    ReadModelFrom = None

    print(modelState)
    paramList = {'FN': FN_opt, 'CN': CN_opt, 'VN': VN_opt}

    modelList = [FN, CN, _VF_]
    modelNameList = ['FN', 'CN', 'VN']
    lossNameList = ['Ptrain', 'Ntrain', 'CLStrain', 'NLL', 'Log10(wLoss)', 'sparseError',
                    'valid-AUC',
                    'test-AUC']

    global check
    check = ul.check(modellist=modelList, modelNameList=modelNameList, lossNameList=lossNameList, optimDic=paramList,
                     checkfolder=checkfolder, cuda=cuda, epoch=e_a, subp=4)

    print('looping ....')
    for epoc in range(epochs):
        print(checkfolder)
        torch.manual_seed(np.random.randint(1000000))
        trainset.initPerm()

        epoch = epoc + e_a if e_a is not None else epoc
        print('===>Training epoch {}'.format(epoch))
        trainLoss = train(modelList, paramList, train_loader, epoch=epoch)
        # print(epoch)
        print('===>Validing epoch {}'.format(epoch))
        validError = testAUC(modelList, vali_loader, epoch=epoch)
        print('===>Testing epoch {}'.format(epoch))
        testError = testAUC(modelList, test_loader, epoch=epoch)

        check.updateloss(
            loss_1Dlist=[trainLoss[0], trainLoss[1], trainLoss[2], trainLoss[3], trainLoss[4], trainLoss[5],
                         validError,
                         testError], train=1, epoch=epoch)
        check.plot_loss(check.loss_history, lossNameList, title='TrianLoss', show=False)

        if epoch % 5 == 0:
            check.save_model(epoch=epoch)
        if validError > bestP:
            bestP = validError
            check.save_model(epoch=epoch * 10000 + np.int(bestP * 10000))

    return modelList


modelStates = []

# 0
modelState = {'epoch_Add': 0,
              'FN_lr': 1e-4,
              'CN_lr': 1e-4,
              'step': 2,  # should 2
              'epn': 41,
              'OEGQDDL3': False,
              'oegFolder': '{}/modelfolder/{}_Ep-{}.pth',
              'genNum': 0
              }
modelStates.append(modelState)
# 1
modelState = {'epoch_Add': modelStates[-1]['epoch_Add'] + modelStates[-1]['epn'],
              'FN_lr': 1e-4,
              'CN_lr': 1e-4,
              'step': 3,
              'epn': 300 - modelStates[-1]['epoch_Add'] - modelStates[-1]['epn'] + 1,
              'OEGQDDL3': True,
              'oegFolder': '{}/modelfolder/{}_Ep-{}.pth',
              'genNum': modelStates[-1]['epoch_Add'] + modelStates[-1]['epn'] - 1
              }
modelStates.append(modelState)

trl, tvl, tel = load_data()
for i in range(0, 2):
    bestP = 0
    epochs = modelStates[i]['epn']
    _step_ = modelStates[i]['step']
    clfer = generate_model(trl, tel, tvl, modelState=modelStates[i])
