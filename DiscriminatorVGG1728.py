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
'''
debug critical 20181004
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models as models
from torch.autograd import Variable
import ResModuleLekyReLU as rm
import util as ul

# DB
def getVggFe():
    vgg11 = models.vgg11(pretrained=True)
    return vgg11.features



class DisNet(nn.Module):
    def __init__(self,z_dim=131,N=(10000,1000,100),dp=0.2,C=1,GN=0,pDp = 0):
        super(DisNet,self).__init__()
        l_name = 'lin{}'
        d_name = 'dropout{}'
        gn_name = 'gn{}'
        r_name = 'relu{}'
        self.lins = nn.Sequential()
        self.GN = GN
        self.dp = dp
        self.pDp = pDp
        if self.pDp !=0:
            self.lins.add_module('pre_dropout', nn.Dropout(p=self.pDp))
        if N:
            self.lins.add_module(l_name.format(0), nn.Linear(z_dim,N[0]))
            if self.GN != 0:
                self.lins.add_module(gn_name.format(0),nn.GroupNorm(self.GN,N[0]))
            if self.dp != 0:   
                self.lins.add_module(d_name.format(0),nn.Dropout(p=dp))
            self.lins.add_module(r_name.format(0),nn.LeakyReLU(0.2))
            for i in range(len(N)-1):
                self.lins.add_module(l_name.format(i+1), nn.Linear(N[i],N[i+1]))
                if self.GN != 0:
                    self.lins.add_module(gn_name.format(i+1),nn.GroupNorm(self.GN,N[i+1]))
                    
                if self.dp != 0:  
                    self.lins.add_module(d_name.format(i+1),nn.Dropout(p=dp))
                self.lins.add_module(r_name.format(i+1),nn.LeakyReLU(0.2))
            
            self.lins.add_module(l_name.format(len(N)), nn.Linear(N[-1],C))
        else:
            self.lins.add_module(l_name.format(0), nn.Linear(z_dim,C))
        # ul.initialize_weights(self)
        self.apply(ul.initialize_weights)

    def forward(self,x):
       

        #return F.sigmoid(self.lin3(x))
        return self.lins(x)        

     
class DisNet1DpCov132(nn.Module):
    def __init__(self,z_dim=131,N=(10000,1000,100),C=1,GN=0):
        super(DisNet1DpCov132,self).__init__()
        l_name = 'lin{}'
        d_name = 'dropout{}'
        gn_name = 'gn{}'
        r_name = 'relu{}'
        self.z_dim = z_dim
        self.lins = nn.Sequential()
        self.GN = GN
        
        self.Cov = nn.Sequential()
        
        self.Cov.add_module('cov11', nn.Conv2d(512,512,3,2,1))#use z_dim at last cov
        self.Cov.add_module('ReLU11',nn.ReLU(inplace=True))
        self.Cov.add_module('cov12', nn.Conv2d(512,z_dim,3,1,1))#use z_dim at last cov
        self.Cov.add_module('ReLU12',nn.ReLU(inplace=True))
        self.Cov.add_module('avg1', nn.AvgPool2d(2, 1, 0))
        self.CovMax = nn.MaxPool2d((26,21))
 
        if N:
            self.lins.add_module(l_name.format(0), nn.Linear(z_dim,N[0]))
            if self.GN != 0:
                self.lins.add_module(gn_name.format(0),nn.GroupNorm(self.GN,N[0]))
             
            self.lins.add_module(d_name.format(0),nn.Dropout(p=0.2))
            self.lins.add_module(r_name.format(0),nn.LeakyReLU(0.2,inplace=True))
            for i in range(len(N)-1):
                self.lins.add_module(l_name.format(i+1), nn.Linear(N[i],N[i+1]))
                if self.GN != 0:
                    self.lins.add_module(gn_name.format(i+1),nn.GroupNorm(self.GN,N[i+1]))
                self.lins.add_module(r_name.format(i+1),nn.LeakyReLU(0.2,inplace=True))
            
            self.lins.add_module(l_name.format(len(N)), nn.Linear(N[-1],C))
        else:
            self.lins.add_module(l_name.format(0), nn.Linear(z_dim,C))

        self.apply(ul.initialize_weights)

    def forward(self,x):
       
        x = self.Cov(x)
        sumBefore = torch.sum(x)
        x = self.CovMax(x)
        xn = torch.numel(x)
        sumAfter = torch.sum(x)
        sl = (sumBefore-sumAfter)/xn
        print(x.size())
        x = x.view(-1,self.z_dim)
        #return F.sigmoid(self.lin3(x))
        return self.lins(x),sl 


