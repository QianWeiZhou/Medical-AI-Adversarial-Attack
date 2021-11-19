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
# 7
epTEST = 320
import os
from pycparser.c_ast import ParamList
from numpy.dual import norm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # @UnusedImport
from PIL import Image
import torch  # @UnusedImport
import pickle  # @UnusedImport
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd  # @UnusedImport
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL  # @UnusedImport
import torch.cuda  # @UnusedImport

import DiscriminatorVGG1728 as dis
import unet_model_bear1728 as gen
import util as ul

import torch.utils.data  # @UnusedImport
import torchvision.transforms as tf  # @UnusedImport

import ImageDataset5f1728test as ID
import png  # pip install pypng @UnresolvedImport

# st = 'train'
# st = 'valid'
st = 'test'
# 0
modelStates = []
modelState = {'genNum': epTEST,
              'GN_lr': 0,
              'DN_lr': 0,
              'GN_pass': 1,
              'step': 0,
              'epn': 0,
              'OEGQDDL3': True,
              'oegFolder': '{}/{}_Ep-{}.pth',
              }
modelStates.append(modelState)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # @UndefinedVariable
_VF_ = dis.getVggFe()


if cuda:
    torch.cuda.set_device(0)
    _VF_ = _VF_.cuda()
_step_ = 0  # 1: autoencoder train G, 2: pretrain D, 3, train Gan

checkfolder = './Samples/fakeImages/'
# ====if its not None, first model loading will try to read models from this folder===================
ReadModelFrom = './Samples/posGAN/'
# ReadModelFrom = None
# ============otherwise, it will read from checkfolder.===============================================

params = {
    'save': '{}/imgs'.format(checkfolder),
    #           'save':None,
    'show': False,
    'block': False,

}
os.system('mkdir -p {}/'.format(checkfolder))

print('checkfolder: {}'.format(checkfolder))
print('ReadModelFrom: {}'.format(ReadModelFrom))
print('params: {}'.format(params))
'''

'''
k = 10
p = 2
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 3
imH = 1728
imW = 1408
c_dim = 3  # for discriminate
raw_dim = 1  # for generator

batch_size = 1

FakeImageFolder = '{}/{}PositiveEP{}'.format(checkfolder, st, modelState['genNum'])
os.system('mkdir -p {}'.format(FakeImageFolder))


class FNnet(nn.Module):
    def __init__(self):
        super(FNnet, self).__init__()
        preNet = dis.DisNet1DpCov132(z_dim=2048, N=(8192, 2048, 256), C=1, GN=32)

        self.linSq = preNet.lins
        self.Cov = preNet.Cov
        self.CovMax = preNet.CovMax

        self.linSq.__delitem__(10)

    def forward(self, x):
        x = self.Cov(x)
        x = self.CovMax(x)
        x = x.view(-1, 2048)
        return self.linSq(x)


def load_data():
    print('loading data!')
    trainset = ID.ImageDataset(transform=None, Posi=1,
                               rootFolder='./Samples/realImages',
                               gn=0, randNeg=False)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=False, drop_last=False, **kwargs)

    return train_loader


def test(modelList, data_loader, epoch=0, modelState=None):
    print('===>testing epoch {}'.format(epoch))

    for ml in modelList:
        ml.train()

    GN = modelList[0]
    DN = modelList[1]
    VN = modelList[2]
    FN = modelList[3]

    iter_count = 0

    VN.eval()
    GN.eval()
    DN.eval()
    FN.eval()
    with torch.no_grad():
        for pX, nX, pT, nT in data_loader:

            iter_count += 1

            ul.printProgressBar(iter_count, data_loader.__len__());

            nX = nX[:, [0], :, :]

            nX, pX = Variable(nX, requires_grad=True), Variable(pX, requires_grad=True)
            if cuda:
                nX, pX = nX.cuda(), pX.cuda()

            fX = GN(nX)

            fX = torch.sigmoid(fX)  # @UndefinedVariable

            saveImages16(fX.data.cpu().numpy(), nT.data.cpu().numpy())

    return None


def saveImages16(Xs, IDs, epoch=0):
    for i in range(Xs.shape[0]):
        X = Xs[i]
        ID = IDs[i]

        X = X[0]
        X = (X * 65535).astype(np.uint16)
        imagePath = '{}/{}fakePos.png'.format(FakeImageFolder, ID)
        with open(imagePath, 'wb') as f:
            writer = png.Writer(width=X.shape[1], height=X.shape[0], bitdepth=16, greyscale=True)
            zgray2list = X.tolist()
            writer.write(f, zgray2list)


def generate_model(train_loader, modelState):
    global ReadModelFrom
    global _VF_

    print(modelState)

    e_a = modelState['genNum']

    print('define models')

    GN = gen.UNet(raw_dim, raw_dim)
    FN = FNnet()
    DN = dis.DisNet(z_dim=256, N=None, dp=0, C=1, GN=0, pDp=0)

    if not isinstance(FN, nn.DataParallel):
        FN = nn.DataParallel(FN)
    if not isinstance(DN, nn.DataParallel):
        DN = nn.DataParallel(DN)
    if not isinstance(_VF_, nn.DataParallel):
        _VF_ = nn.DataParallel(_VF_)
    if not isinstance(GN, nn.DataParallel):
        GN = nn.DataParallel(GN)

    if modelState['OEGQDDL3']:
        print('loading models ....')

        if ReadModelFrom:
            print('reading model from {}'.format(ReadModelFrom))
        else:
            ReadModelFrom = checkfolder
        oegFolder = modelState['oegFolder']
        genNum = modelState['genNum']
        GN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom, 'GN', genNum), map_location=torch.device('cpu')))
        try:
            DN.load_state_dict(
                torch.load(oegFolder.format(ReadModelFrom, 'DN', genNum), map_location=torch.device('cpu')))
        except:
            print('====> no DN history found! use default initialization.')
        try:
            FN.load_state_dict(
                torch.load(oegFolder.format(ReadModelFrom, 'FN', genNum), map_location=torch.device('cpu')))
        except:
            print('====> no FN history found! use default initialization.')
        try:
            _VF_.cpu()
            _VF_.load_state_dict(
                torch.load(oegFolder.format(ReadModelFrom, 'VN', genNum), map_location=torch.device('cpu')))
        except:
            print('====> no VN history found! use default initialization.')

    #         ReadModelFrom = None

    if cuda:
        print('move to GPU')
        GN.cuda()
        DN.cuda()
        FN.cuda()
        _VF_.cuda()

    ReadModelFrom = None
    
    modelList = [GN, DN, _VF_, FN]
    modelNameList = ['GN', 'DN', 'VN', 'FN']

    lossNameList = ['d_Drf_loss_a', 'd_Dff_loss_a', 'Fn-Disf_a', 'r-f_loss_a', 'Xnoise_l2', 'w_loss_a']

    epoch = 0 + e_a if e_a is not None else 0

    print(checkfolder)
    #         print()
    with torch.no_grad():
        trainLoss = test(modelList, train_loader, epoch=epoch, modelState=modelState)

    return modelList


trl = load_data()

for i in range(0, 1):  # len(modelStates)):
    epochs = modelStates[i]['genNum']
    _step_ = modelStates[i]['step']
    clfer = generate_model(trl, modelState=modelStates[i])
