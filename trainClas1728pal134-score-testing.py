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
'''6

'''
import os
from pycparser.c_ast import ParamList
from numpy.dual import norm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"# if use '2,1', then in pytorch, gpu2 has id 0
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL

import DiscriminatorVGG1728 as dis  
import unet_model_bear1728 as gen
import util as ul

import ImageDataset4f1728sco as ID #use flip label
import torch.cuda
import torch.utils.data
import focalloss 
import torchvision.transforms as tf

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # @UndefinedVariable
_VF_ = dis.getVggFe()
# _VF_.load_state_dict(torch.load('./../checkfolderV2-50-step1234/modelfolder/VN_Ep-400.pth',map_location=torch.device('cpu')))

if cuda:
    _VF_ = _VF_.cuda()
     
fl_mm = 0 
fl_weight = np.array([1/2,1/2])
# fl_weight = np.array([0.27,  0.36,  0.36])#27
print('fl_weight = {}'.format(fl_weight.tolist())) 
fl = focalloss.FocalLoss(gamma=2,alpha=fl_weight.tolist())

AUCepoch = 788122

settypeSTR = 'test'
settype = 0

DataFolder = './Samples/realImages/' 
checkfolder = './Samples/{}DataScores-{}/'.format(settypeSTR,AUCepoch)

#====if its not None, first model loading will try to read models from this folder===================
ReadModelFrom = './Samples/Classifier'
# ReadModelFrom = None
#============otherwise, it will read from checkfolder.===============================================
os.system('mkdir -p {}/'.format(checkfolder))

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 2
imH = 1728
imW = 1408
c_dim = 3
raw_dim = 1 # for generator
k=10
p=4 
batch_size = 1



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

    testset = ID.ImageDataset(transform=None,setType=settype,rootFolder=DataFolder)
    test_loader = torch.utils.data.DataLoader(testset,
                                        batch_size=batch_size,
                                        shuffle=False,drop_last=False,
                                        **kwargs) 
    

    return test_loader

def test(modelList,data_loader,epoch=0): 
    FN = modelList[0]
    CN = modelList[1]
    
    
    FN.eval()
    CN.eval()
    _VF_.eval()
    
    iter_count = 0
    Scores = None
    totalPtarget = np.array([0,0])
    for batch_index, batch_data in enumerate(data_loader,0):
        iter_count += 1.0
        ul.printProgressBar(iter_count,data_loader.__len__());
        
        cX,cT,cID = batch_data
        cX, cT = Variable(cX),Variable(cT)

        if cuda:
            cX,cT = cX.cuda(),cT.cuda()
        
        vcX = _VF_(cX)
        fFeature = FN(vcX)
        output = CN(fFeature)
        output = F.softmax(output, dim=1)
        output = np.float32(output.data.cpu().numpy())
        ids = np.expand_dims(np.float32(cID.cpu().numpy()),axis=1)
        lbs = np.expand_dims(np.float32(cT.data.cpu().numpy()),axis=1)
        tempNp = np.concatenate((ids,output,lbs),axis=1)
        if iter_count <= 1:
            Scores = tempNp
        else:
            Scores = np.concatenate((Scores,tempNp),axis=0) 
        
    return Scores 


def saveScores(Scores,title):
    np.savetxt('{}/Scores-{}-real.txt'.format(checkfolder,title),Scores,fmt='%10.2f',header='%6s,%6s,%6s,%6s'%('id','pos','neg','label'))
 
def getAUC(TRs): 
    TRs = np.flip(TRs,1)  
    temp = np.zeros(1,TRs.shape[1]) 
    temp[0,1:] = TRs[1,:-1]
    temp = TRs[1,:]-temp[0,:]
    AUC = np.sum(TRs[0,:]*temp[0,:])  
    return AUC 

def generate_model(test_loader):
    global ReadModelFrom
    global AUCepoch
    global _VF_
    global settypeSTR

    print('define models')

    FN = FNnet()

    CN = dis.DisNet(z_dim=256,N=None,dp=0,C=2,GN=0,pDp=0)

    if not isinstance(FN, nn.DataParallel):
        FN = nn.DataParallel(FN)
    if not isinstance(CN, nn.DataParallel):
        CN = nn.DataParallel(CN)
    if not isinstance(_VF_, nn.DataParallel):
        _VF_ = nn.DataParallel(_VF_)
   
    
    print('loading models ....')
    if ReadModelFrom:
        print('reading model from {}'.format(ReadModelFrom))
    else:
        ReadModelFrom = checkfolder
    oegFolder = '{}/{}_Ep-{}.pth'
    genNum = AUCepoch
    print(oegFolder.format(ReadModelFrom,'FN',genNum))
    CN.load_state_dict(torch.load(oegFolder.format(ReadModelFrom,'CN',genNum),map_location=torch.device('cpu')))
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
        FN.cuda()
        CN.cuda()
        _VF_.cuda()
    print(checkfolder) 
     
    modelList = [FN,CN,_VF_]
    epoch = AUCepoch
    print('===>Testing epoch {}'.format(epoch))
    testTRs = test(modelList,test_loader,epoch=epoch)
    
    saveScores(testTRs,settypeSTR)
        

    return modelList
        
 
tsl = load_data()
clfer = generate_model(tsl)
