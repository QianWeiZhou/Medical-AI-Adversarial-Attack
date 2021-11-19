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
#22

import os
#from pycparser.c_ast import ParamList
from numpy.dual import norm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import torch.cuda

import DiscriminatorVGG1728 as dis
import unet_model_bear1728 as gen
import util as ul
import ImageDataset5f1728 as ID 

import torch.utils.data
import torchvision.transforms as tf

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # @UndefinedVariable
_VF_ = dis.getVggFe()


if cuda:
    torch.cuda.set_device(0)
    _VF_ = _VF_.cuda()

_step_ = 0 

DataFolder = './Samples/realImages/' 
checkfolder = './checkfolder-negativeLookingGenerator/'
#====if its not None, first model loading will try to read models from this folder===================
preTrainedModel = './checkfolder-Classifier/modelfolder/{}_Ep-0.pth' # change this one to your classifier.
ReadModelFrom = None
#============otherwise, it will read from checkfolder.===============================================

params = {
          'save':'{}/imgs'.format(checkfolder),
          'show':False,
          'block':False,
          'PosiID':[1049,1050,1095],
          'NegaID':[3917,]
          }
os.system('mkdir -p {}/'.format(checkfolder))
if params['save']:
    os.system('mkdir -p {}'.format(params['save']))
    
print('checkfolder: {}'.format(checkfolder))
print('ReadModelFrom: {}'.format(ReadModelFrom))
print('params: {}'.format(params))
'''

'''
k=10
p=2
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {} 
n_classes = 3
imH = 1728
imW = 1408
c_dim = 3 # for discriminate
raw_dim = 1 # for generator

batch_size = 2 # should be 2

pretransform = tf.Compose([
            tf.RandomVerticalFlip(),
            tf.RandomAffine(45, translate=(0,0), scale=None, shear=45, resample=PIL.Image.BILINEAR, fillcolor=0),#-D5
            ])



class FNnet(nn.Module):
    def __init__(self):
        super(FNnet,self).__init__()
        preNet = dis.DisNet1DpCov132(z_dim=2048,N=(8192,2048,256),C=1,GN=32)
        
        self.linSq = preNet.lins
        self.Cov = preNet.Cov
        self.CovMax = preNet.CovMax

        self.linSq.__delitem__(10)
        
    def forward(self,x):
        x = self.Cov(x)
        x = self.CovMax(x)
        x = x.view(-1,2048)
        return self.linSq(x)

def load_data():
    print('loading data!')
    trainset = ID.ImageDataset(transform=pretransform,Posi=1,
                               rootFolder=DataFolder,
                               gn=0)
    
    train_loader = torch.utils.data.DataLoader(trainset,
                                        batch_size=batch_size,
                                        shuffle=True,drop_last=True, **kwargs)

    return train_loader

def lossF(x):
    return -torch.log(1+torch.exp(-x))  # @UndefinedVariable

batchMean = torch.zeros(batch_size,c_dim,imH,imW).float()  # @UndefinedVariable
batchMean[:,0,:,:] = batchMean[:,0,:,:]+0.485
batchMean[:,1,:,:] = batchMean[:,1,:,:]+0.456
batchMean[:,2,:,:] = batchMean[:,2,:,:]+0.406
batchStd = torch.zeros(batch_size,c_dim,imH,imW).float()  # @UndefinedVariable
batchStd[:,0,:,:] = batchStd[:,0,:,:]+0.229
batchStd[:,1,:,:] = batchStd[:,1,:,:]+0.224
batchStd[:,2,:,:] = batchStd[:,2,:,:]+0.225
if cuda:
    batchMean = batchMean.cuda()
    batchStd = batchStd.cuda()

def vfOutCat(x):
    x = F.sigmoid(x)
    x1 = x
    x2 = x
    x3 = x
    x = torch.cat((x1,x2,x3),dim=1)  # @UndefinedVariable
    x = (x-batchMean)/batchStd
    
    return x

def vfOut1Cat(x):
    x = F.sigmoid(x)
    x1 = x
    x2 = x
    x3 = x
    x = torch.cat((x1,x2,x3),dim=1)  # @UndefinedVariable
    x = (x-batchMean[[0]])/batchStd[[0]]
    
    return x


def train(modelList,paramList,data_loader,epoch=0,modelState=None):
    global k
    global p
    global _step_
    paramPID = params['PosiID'].copy()
    paramNID = params['NegaID'].copy()
    print('===>training epoch {}'.format(epoch))
    
    for ml in modelList:
        ml.train()

    GN = modelList[0]
    DN = modelList[1]
    VN = modelList[2]
    FN = modelList[3]

    iter_count = 0
    
    d_Drf_loss_a = 0
    d_Dff_loss_a = 0
    d_r_f_loss_a = 0
    w_loss_a = 0 
    fe_Disf_a = 0   
    fe_l2Loss_a = 0
    fnCount = 0.0
    
    

    for pX,nX,pT,nT in data_loader:
        l1T = Variable(torch.zeros(batch_size,raw_dim,imH,imW))  # @UndefinedVariable

            
        iter_count += 1     
        ul.printProgressBar(iter_count,data_loader.__len__());

        nX, pX = Variable(nX,requires_grad=True), Variable(pX)
        
        pX = pX[:,[0],:,:]
        
        if cuda:
            nX, pX, l1T = nX.cuda() ,pX.cuda(), l1T.cuda()
        
        rX = nX
        
        for ml in modelList:
            ml.train()
            
        paramList['GN'].zero_grad()
        paramList['DN'].zero_grad()
        paramList['FN'].zero_grad()
        paramList['VN'].zero_grad()  
            
        #######################
        #Discriminator
        #######################

        if _step_//10 >= 1 :
            with torch.no_grad():
                fX = GN(pX)
                fX = vfOutCat(fX)

            # Real images
            vfrX = VN(rX)
            rFeature = FN(vfrX)
            real_validity = DN(rFeature)
            # Fake images
            
            sampleDiv = None
            sampleDiv_validity = real_validity
           
            sampleDiv = rX

            sampleDiv_grad_out = Variable(Tensor(sampleDiv.size(0), 1).fill_(1.0),requires_grad=False)
            sampleDiv_grad = autograd.grad(sampleDiv_validity,
                                      sampleDiv,
                                      sampleDiv_grad_out,
                                      create_graph=True,
                           retain_graph=True,
                      only_inputs=True)[0]
            sampleDiv_grad_norm = sampleDiv_grad.view(sampleDiv_grad.size(0),-1).pow(2).sum(1)**(p)
    
            div_gp = torch.mean(sampleDiv_grad_norm) * k / 2  # @UndefinedVariable
            
            # Adversarial loss
            vffX = VN(fX)
            fFeature = FN(vffX)
            fake_validity = DN(fFeature)
            
            Drf = -torch.mean(lossF(real_validity))  # @UndefinedVariable
            Dff = torch.mean(lossF(fake_validity))  # @UndefinedVariable
            Ddis = Drf + Dff
            d_loss = Ddis + div_gp + torch.abs(Drf-1) #extra Drf to keep Drf close to 1. @UndefinedVariable
            
            d_loss.backward()
            if (_step_//10)%10 == 1:
                paramList['DN'].step()
            if (_step_//100)%10 == 1:
                paramList['FN'].step()
            if (_step_//1000)%10 == 1:
                paramList['VN'].step()
    
            paramList['GN'].zero_grad()
            paramList['DN'].zero_grad()
            paramList['VN'].zero_grad()
            paramList['FN'].zero_grad()
            
            d_Drf_loss_a += -Drf.cpu().data.numpy()
            d_Dff_loss_a += Dff.cpu().data.numpy() 
            d_r_f_loss_a += -Ddis.cpu().data.numpy()   
            w_loss_a += div_gp.cpu().data.numpy()    
                
        #######################
        #Feature
        #######################
        if (iter_count+epoch)%modelState['GN_pass'] == 0 and ( _step_%10 >= 1 ): 

            fnCount += 1.0
            
            fX = GN(pX)
            fX = vfOutCat(fX)
            Xnoise = fX[:,[0],:,:] - pX 
            XnoiseM = torch.mean(pX**2)  # @UndefinedVariable
            
            g_loss = 0
            l2_Xnoise = 0
            
            if _step_%10==2 :
                vffX = _VF_(fX)
                fFeature = FN(vffX)
                fake_validity = DN(fFeature)
                g_loss = -torch.mean(lossF(fake_validity))  # @UndefinedVariable

                l2_Xnoise = F.mse_loss(Xnoise,l1T)/XnoiseM
                

                loss = g_loss
                loss = loss + l2_Xnoise
            
                loss.backward()
                paramList['GN'].step()
                fe_Disf_a += -g_loss.cpu().data.numpy()
                
            if _step_%10 == 1:
                
                l2_Xnoise = F.mse_loss(Xnoise,l1T)/XnoiseM
                
                loss = l2_Xnoise 
            
                loss.backward()
                paramList['GN'].step()
                fe_Disf_a += 0.0
 
            fe_l2Loss_a += l2_Xnoise.cpu().data.numpy()
            
        if (epoch) % 5  == 0 or epoch == 1:
            if len(paramPID) != 0:
                temp = paramPID.copy()
                for i in range(len(temp)):
                    pID = getBatchID(temp[i], pT)
                    if pID != -1:
                        saveImages(pX[pID,[0],:,:],modelList,ID=temp[i],Posi=True,epoch=epoch)
                        paramPID.remove(temp[i])
                    
            if len(paramNID) != 0:
                temp = paramNID.copy()
                for i in range(len(temp)):
                    nID = getBatchID(temp[i], nT)
                    if nID != -1:
                        saveImages(nX[nID,[0],:,:],modelList,ID=temp[i],Posi=False,epoch=epoch)
                        paramNID.remove(temp[i])
            
        
    d_Drf_loss_a = d_Drf_loss_a/iter_count
    d_Dff_loss_a = d_Dff_loss_a/iter_count
    d_r_f_loss_a = d_r_f_loss_a/iter_count
    w_loss_a = w_loss_a/iter_count
    
    if fnCount != 0:
        fe_Disf_a = fe_Disf_a/fnCount
        fe_l2Loss_a = fe_l2Loss_a/fnCount
    

    return d_Drf_loss_a,d_Dff_loss_a,fe_Disf_a,d_r_f_loss_a,fe_l2Loss_a, w_loss_a

def getBatchID(gID,batchIDs):
    idgb = (batchIDs==gID)

    if idgb.sum() != 0:
        idgb = idgb.nonzero()
        idgb = idgb[0][0]
    else:
        idgb = -1
    return idgb

def saveImages(X,modelList,ID=0,Posi=None,epoch=0):

    GN = modelList[0]
    DN = modelList[1]
    VN = modelList[2]
    FN = modelList[3]
    
    DN.eval()
    GN.eval()
    VN.eval()
    FN.eval()

    X=X.view(1,raw_dim,imH,imW)
    X3c = X.clone()
    X3c = X3c*0.229-0.485
    X3c = vfOut1Cat(X3c)

    vfX = VN(X3c)
    fFeature = FN(vfX)
    rv = DN(fFeature)
    
    fX = GN(X)
#     fX = F.tanh(fX)*2.64
    fX = vfOut1Cat(fX)
    Xnoise = fX[:,[0],:,:] - X 
    
    vffX = _VF_(fX)
    ffFeature = FN(vffX)
    fv = DN(ffFeature)
    
    plt.figure(1,figsize=(15,5))
    plt.clf()
    ax = plt.subplot(131)
    img = np.array(X.data.cpu().tolist()).reshape(raw_dim,imH, imW)
    imgs = img[0,:,:]
    ax.imshow(imgs,cmap=plt.get_cmap('gray'))
    ax.set_title('X:{:.2e};ID:{};Posi:{};Max:{:.2e};Min:{:.2e}'.format(rv.data.cpu().numpy()[0][0],ID,Posi,imgs.max(),imgs.min()))
    
    
    ax = plt.subplot(132)
    img = np.array(fX.data.cpu().tolist()).reshape(c_dim,imH, imW)
    imgs = img[0,:,:]
    ax.imshow(imgs,cmap=plt.get_cmap('gray'))
    ax.set_title('fX:{:.2e};Max:{:.2e};Min:{:.2e}'.format(fv.data.cpu().numpy()[0][0],imgs.max(),imgs.min()))
    
    
    ax = plt.subplot(133)
    img = np.array(Xnoise.data.cpu().tolist()).reshape(raw_dim,imH, imW)
    imgs = img[0,:,:]
    ax.imshow(np.abs(imgs),cmap=plt.get_cmap('gray'))
    ax.set_title('Xnoise;Max{:.2e};Min{:.2e}'.format(imgs.max(),imgs.min()))
    
    
    if params['save'] is not None:
        plt.savefig(os.path.join(params['save'],'X-fX-Xnoise-P:{}-ID:{}-ep:{}.png'.format(Posi,ID,epoch)),dpi=300)
    
    if params['show']:
        plt.show(block=params['block'])
    

def generate_model(train_loader,modelState):
    global ReadModelFrom
    global _VF_
    global preTrainedModel
    
    print(modelState)
    
    e_a = modelState['epoch_Add']

    print('define models')
    

    GN = gen.UNet(raw_dim,raw_dim)
    
    FN = FNnet()
    DN = dis.DisNet(z_dim=256,N=None,dp=0,C=1,GN=0,pDp=0)

    if not isinstance(FN, nn.DataParallel):
        FN = nn.DataParallel(FN)
    if not isinstance(DN, nn.DataParallel):
        DN = nn.DataParallel(DN)
    if not isinstance(_VF_, nn.DataParallel):
        _VF_ = nn.DataParallel(_VF_)
    if not isinstance(GN, nn.DataParallel):
        GN = nn.DataParallel(GN)

    if preTrainedModel:
        try:
            FN.load_state_dict(torch.load(preTrainedModel.format('FN'),map_location=torch.device('cpu')))
        except:
            print('====> no FN Pre found!')
        try:
            _VF_.cpu()
            _VF_.load_state_dict(torch.load(preTrainedModel.format('VN'),map_location=torch.device('cpu')))
        except:
            print('====> no VN Pre found!')
        preTrainedModel = None
    else:
        print('====> no Need to load Pretrained model')

    if modelState['OEGQDDL3'] :
        print('loading models ....')

        if ReadModelFrom:
            print('reading model from {}'.format(ReadModelFrom))
        else:
            ReadModelFrom = checkfolder
        oegFolder = modelState['oegFolder']
        genNum = modelState['genNum']
        try:
            GN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom,'GN',genNum),map_location=torch.device('cpu')))
        except:
            print('====> no GN history found! use default initialization.')
        try:
            DN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom,'DN',genNum),map_location=torch.device('cpu')))
        except:
            print('====> no DN history found! use default initialization.')
        try:
            FN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom,'FN',genNum),map_location=torch.device('cpu')))
        except:
            print('====> no FN history found! use default initialization.')
        try:
            _VF_.cpu()
            _VF_.load_state_dict(torch.load(oegFolder.format(ReadModelFrom,'VN',genNum),map_location=torch.device('cpu')))
        except:
            print('====> no VN history found! use default initialization.')
        

    print('move to GPU')
    if cuda:
        GN.cuda()
        FN.cuda()
        DN.cuda()
        _VF_.cuda()

    GN_lr = modelState['GN_lr']
    DN_lr = modelState['DN_lr']
    FN_lr = modelState['DN_lr']
    
    GN_opt = optim.Adam(GN.parameters(), lr=GN_lr,betas=(0.5,0.9))
    DN_opt = optim.Adam(DN.parameters(), lr=DN_lr,betas=(0.5,0.9))
    FN_opt = optim.Adam(FN.parameters(), lr=FN_lr,betas=(0.5,0.9))
    VN_opt = optim.Adam(_VF_.parameters(),lr=DN_lr/10,betas=(0.5,0.9))
    
    if modelState['OEGQDDL3'] :
        try:
            GN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom,'GN',modelState['genNum']),map_location=torch.device('cpu'))
            GN_opt.load_state_dict(GN_opt_st)
        except:
            print('====> no GN_optim history found! use default initialization.')    
        try:    
            DN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom,'DN',modelState['genNum']),map_location=torch.device('cpu'))
            DN_opt.load_state_dict(DN_opt_st)
        except:
            print('====> no DN_optim history found! use default initialization.')    
        try:    
            FN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom,'FN',modelState['genNum']),map_location=torch.device('cpu'))
            FN_opt.load_state_dict(FN_opt_st)
        except:
            print('====> no FN_optim history found! use default initialization.')     
        try:    
            VN_opt_st = torch.load('{}/modelfolder/{}_OpSt_Ep-{}.pth'.format(ReadModelFrom,'VN',modelState['genNum']),map_location=torch.device('cpu'))
            VN_opt.load_state_dict(VN_opt_st)
        except:
            print('====> no VN_optim history found! use default initialization.')   
            
        for param_group in GN_opt.param_groups:
            param_group['lr'] = GN_lr
        for param_group in DN_opt.param_groups:
            param_group['lr'] = DN_lr
        for param_group in FN_opt.param_groups:
            param_group['lr'] = FN_lr
        for param_group in VN_opt.param_groups:
            param_group['lr'] = DN_lr/10
        
            
    ReadModelFrom = None
    
    paramList = {'GN':GN_opt,'DN':DN_opt,'VN':VN_opt,'FN':FN_opt}

    modelList = [GN,DN,_VF_,FN]
    modelNameList = ['GN','DN','VN','FN']
    
    
    lossNameList = ['d_Drf_loss_a','d_Dff_loss_a','Fn-Disf_a','r-f_loss_a','Xnoise_l2','w_loss_a']

    check = ul.check(modellist=modelList,modelNameList=modelNameList,lossNameList=lossNameList,optimDic=paramList,
                     checkfolder = checkfolder,cuda=cuda,epoch=e_a,subp=3)
    
    print(modelState)
    print('looping ....')
    for epoc in range(epochs):
        torch.manual_seed(np.random.randint(1000000))
        epoch = epoc+e_a if e_a is not None else epoc
        print(checkfolder)
        trainLoss = train(modelList,paramList,train_loader,epoch=epoch,modelState=modelState)
        
        logLoss = np.zeros(len(trainLoss))
        for i in range(len(trainLoss)):
            logLoss[i] = trainLoss[i]
        
        check.updateloss(loss_1Dlist=[logLoss[0],logLoss[1],logLoss[2],logLoss[3],logLoss[4],logLoss[5]],train=1,epoch=epoch)
        check.plot_loss(check.loss_history,lossNameList,title='TrianLoss',show=False)

        if (epoch % 5 == 0 and _step_ <12) or epoch % 10 == 0:
            check.save_model(epoch=epoch)

    
    return modelList


modelStates = []
#'VN','FN','DN','GN'
#  1    1    1    1(mse) 2(gan)
#0
modelState = {'epoch_Add':0,
                'GN_lr':1e-4,
                'DN_lr':1e-4,
                'GN_pass':1,
                'step':1, #should be 1
                'epn': 11,
                'OEGQDDL3':False,
                'oegFolder':'{}/modelfolder/{}_Ep-{}.pth',
                'genNum':0
                }
modelStates.append(modelState)
#1
modelState = {'epoch_Add':modelStates[-1]['epoch_Add']+modelStates[-1]['epn'],
                'GN_lr':1e-5,
                'DN_lr':1e-4,
                'GN_pass':5,
                'step':10,
                'epn': 20-modelStates[-1]['epoch_Add']-modelStates[-1]['epn']+1,
                'OEGQDDL3':True,
                'oegFolder':'{}/modelfolder/{}_Ep-{}.pth',
                'genNum':modelStates[-1]['epoch_Add']+modelStates[-1]['epn']-1
                }
modelStates.append(modelState)
#2
modelState = {'epoch_Add':modelStates[-1]['epoch_Add']+modelStates[-1]['epn'],
                'GN_lr':1e-5,
                'DN_lr':1e-4,
                'GN_pass':5,
                'step':12,
                'epn': 30-modelStates[-1]['epoch_Add']-modelStates[-1]['epn']+1,
                'OEGQDDL3':True,
                'oegFolder':'{}/modelfolder/{}_Ep-{}.pth',
                'genNum':modelStates[-1]['epoch_Add']+modelStates[-1]['epn']-1
                }
modelStates.append(modelState)
#3
modelState = {'epoch_Add':modelStates[-1]['epoch_Add']+modelStates[-1]['epn'],
                'GN_lr':1e-5,
                'DN_lr':1e-4,
                'GN_pass':5,
                'step':112,
                'epn': 330-modelStates[-1]['epoch_Add']-modelStates[-1]['epn']+1,
                'OEGQDDL3':True,
                'oegFolder':'{}/modelfolder/{}_Ep-{}.pth',
                'genNum':modelStates[-1]['epoch_Add']+modelStates[-1]['epn']-1
                }
modelStates.append(modelState)
#4
modelState = {'epoch_Add':modelStates[-1]['epoch_Add']+modelStates[-1]['epn'],
                'GN_lr':1e-5,
                'DN_lr':1e-4,
                'GN_pass':5,
                'step':1112,
                'epn': 600-modelStates[-1]['epoch_Add']-modelStates[-1]['epn']+1,
                'OEGQDDL3':True,
                'oegFolder':'{}/modelfolder/{}_Ep-{}.pth',
                'genNum':modelStates[-1]['epoch_Add']+modelStates[-1]['epn']-1
                }
modelStates.append(modelState)




trl = load_data()

for i in range(0,5):#len(modelStates)):
    epochs = modelStates[i]['epn']
    _step_= modelStates[i]['step']
    clfer = generate_model(trl,modelState=modelStates[i])

