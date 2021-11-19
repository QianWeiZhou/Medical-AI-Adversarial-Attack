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
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import time
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import sys
import copy



# Backbone of the model
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def dconv3x3(in_planes, out_planes, stride=2, padding=1, output_padding=1):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, 3, stride=stride,
                              padding=padding, output_padding=output_padding, bias=True);


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def bn2d(inplanes, eps=1e-5, mode=0):
    '''
    mode=0: use batchnorm
    mode=1: use layernorm
    mode=2: do nothing
    '''
    if mode == 0:
        return nn.BatchNorm2d(inplanes, eps=eps)
    elif mode == 1:
        return LayerNorm(inplanes, eps=eps)
    elif mode == 2:
        return Identity()
    elif mode == 3:
        return CorlorNorm(inplanes, eps=eps)
    elif mode == 4:
        return nn.GroupNorm(8, inplanes, eps=eps)
    elif mode == 5:
        return pixelwise_norm_layer()
    elif mode == 6:
        return nn.GroupNorm(32, inplanes, eps=eps)
    elif mode == 7:
        return nn.GroupNorm(32, inplanes, eps=1e-10)


class pixelwise_norm_layer(nn.Module):
    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5  # @UndefinedVariable



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # @UndefinedVariable
        self.beta = nn.Parameter(torch.zeros(features))  # @UndefinedVariable
        self.eps = eps

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class CorlorNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # @UndefinedVariable
        self.beta = nn.Parameter(torch.zeros(features))  # @UndefinedVariable
        self.eps = eps
        self.features = features

    def forward(self, x):
        shape = [-1] + [self.features] + [1] * (x.dim() - 2)
        mean = x.view(x.size(0), x.size(1), -1).mean(2).view(*shape)
        std = x.view(x.size(0), x.size(1), -1).std(2).view(*shape)

        y = (x - mean) / (std + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


def initialize_weights(m):
    """Initialize model weights.
    """
    # for m in model.modules():
    if isinstance(m, nn.Conv2d):

        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            m.bias.data.zero_()



class check():
    def __init__(self, modellist, modelNameList, lossNameList, optimDic=None, checkfolder='./../checkfolder/',
                 cuda=False, epoch=0, subp=0):
        self.modellist = modellist
        self.modelNameList = modelNameList
        self.lossNameList = lossNameList
        self.optimDic = optimDic
        self.loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.checkfolder = checkfolder
        self.modelfolder = os.path.join(self.checkfolder, 'modelfolder')
        self.logfolder = os.path.join(self.checkfolder, 'log')
        self.count = 0
        self.cuda = cuda
        self.start_epoch = epoch
        self.subp = subp
        if os.path.exists(self.checkfolder) != True:
            os.mkdir(self.checkfolder)

        if os.path.exists(self.modelfolder) != True:
            os.mkdir(self.modelfolder)

        if os.path.exists(self.logfolder) != True:
            os.mkdir(self.logfolder)

    def updateloss(self, loss_1Dlist, train, epoch=0):
        ''' 
        
        train: 1 for train, 2 for validation, 3 for test     
        '''
        if train == 1:
            self.loss_history.append(loss_1Dlist)
            self.report_loss(epoch, loss_1Dlist, prefix='train_loss: ')
        elif train == 2:
            self.val_loss_history.append(loss_1Dlist)
            self.report_loss(epoch, loss_1Dlist, prefix='val_loss: ')
        elif train == 3:
            self.test_loss_history.append(loss_1Dlist)
            self.report_loss(epoch, loss_1Dlist, prefix='test_loss: ')
        else:
            self.log('error train parameter')

    def report_loss(self, epoch, loss_1Dlist, prefix=''):
        '''
        Print loss
        '''
        if self.lossNameList.__len__() != loss_1Dlist.__len__():
            self.log('error, loss number unmatch.')
            return

        s = prefix + 'Epoch-{}; '.format(epoch)
        for i in range(loss_1Dlist.__len__()):
            s += self.lossNameList[i] + ': {:.4}; '.format(loss_1Dlist[i])

        self.count += 1
        self.log(s)

    def log(self, text, array=None):
        """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
        """
        f = open(self.logfolder + '/log.txt', 'a')
        if array is not None:
            text = text.ljust(25)
            text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
                str(array.shape),
                array.min() if array.size else "",
                array.max() if array.size else ""))

        text = time.strftime('\r\n %Y%m%d%H%M%S', time.localtime(time.time())) + '===>' + text
        print(text)
        f.write(text)
        f.close()
        return text

    def save_model(self, epoch):
        for i in range(self.modellist.__len__()):
            p = os.path.join(self.modelfolder, "{}_Ep-{}.pth".format(self.modelNameList[i], epoch))
            torch.save(self.modellist[i].state_dict(), p)

        if self.optimDic:
            for k in self.optimDic:
                p = os.path.join(self.modelfolder, "{}_OpSt_Ep-{}.pth".format(k, epoch))
                v = self.optimDic[k]
                torch.save(v.state_dict(), p)

    def plot_loss(self, loss, labels, title='TrainLoss', save=True, show=True):
        if self.count < 2:
            return

        if self.subp == 0:
            plt.figure(title, figsize=(5, 5))
            plt.gcf().clear()
            lines = plt.plot(loss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(lines, labels)
        else:
            plt.figure(title, figsize=(10, 5))
            plt.gcf().clear()
            npLoss = np.array(loss)

            plt.subplot(1, 2, 1)
            lines1 = plt.plot(npLoss[:, 0:self.subp])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(lines1, labels[0:self.subp])

            plt.subplot(1, 2, 2)
            lines2 = plt.plot(npLoss[:, self.subp:])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(lines2, labels[self.subp:])

        if save:
            save_path = os.path.join(self.logfolder, title + str(self.start_epoch) + ".png")
            plt.savefig(save_path)

        if show:
            plt.show(block=False)
            plt.pause(0.1)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    s = '{} |{}| {}% {}'.format(prefix, bar, percent, suffix)
    sys.stdout.write(' ' * (s.__len__() + 3) + '\r')
    sys.stdout.flush()
    sys.stdout.write(s + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()




