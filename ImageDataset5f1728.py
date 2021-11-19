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
# copy from ImageDataset5f3328.py
import torch
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as tf
import PIL.Image
import re

class ImageDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self,transform=None,Posi=1,rootFolder='../data/',
                 nbChannel = 3, img_mean = (0.485, 0.456, 0.406), 
                 img_std = (0.229, 0.224, 0.225), gn = 0,setType='train',
                 randNeg = True):
        # TODO
        # 1. Initialize file path or list of file names.
        self.rootFolder = rootFolder;
        self.randNeg = randNeg;
        self.nbC = nbChannel
        self.imean = img_mean
        self.istd = img_std
        self.pn = Posi
        self.gn = gn
        self.setType = setType 
        self.negative = ['%s/negb9List-%s.txt'%(self.rootFolder,self.setType)]
        self.positive = ['%s/cancerList-%s.txt'%(self.rootFolder,self.setType)]
        print(self.negative,self.positive)
           
        
        self.regex = "\d+"
        self.posi_list = []
        self.posi_id = []
        self.nega_list = []
        self.nega_id = []

        for i in range(self.positive.__len__()):
            file = open(self.positive[i],'r',encoding='utf-8')  
            line = file.readline()
            while line:
                line = line.strip('\n').strip()
                if line[0] != '#':
                    path, filename = os.path.split(line)
                    temp_id = int(re.search(self.regex, filename).group())

                    fh = '%s/%s'%(self.rootFolder,line)
                    
                    if fh != None:
                        self.posi_id.append(temp_id)
                        self.posi_list.append(fh)
                    else:#maybe lost, to be tested! 20201206
                        print('pass==> {}'.format(line))
                else:
                    print('pass==> {}'.format(line))
#                     self.posi_list.append('pass')
                line = file.readline()

            file.close()
            

        for i in range(self.negative.__len__()):
            file = open(self.negative[i],'r',encoding='utf-8')  
            line = file.readline()
            while line:
                line = line.strip('\n').strip()
                if line[0] != '#':
                    path, filename = os.path.split(line)
                    temp_id = int(re.search(self.regex, filename).group())

                    fh = '%s/%s'%(self.rootFolder,line)
                    
                    if fh != None:
                        self.nega_id.append(temp_id)
                        self.nega_list.append(fh)
                    else:
                        print('pass==> {}'.format(line))
                else:
                    print('pass==> {}'.format(line))
#                     self.nega_list.append(['pass'])
                line = file.readline()

            file.close()

        self.posi_len = self.posi_list.__len__()
        self.nega_len = self.nega_list.__len__()
        
        print(self.posi_len,self.nega_len)
        
        
        if self.pn == 1 or self.pn == 4:
            self.set_len = torch.max(torch.Tensor([self.posi_len,self.nega_len]).int())  # @UndefinedVariable
        elif self.pn == 2:
            self.set_len = self.posi_len
        elif self.pn == 3:
            self.set_len = self.nega_len
        

        self.posi_perm = torch.randperm(self.posi_len)
        self.nega_perm = torch.randperm(self.nega_len)
    
        self.preTransform = transform
        if self.imean:
            self.normTF = tf.Normalize(self.imean,self.istd)
        else:
            self.normTF = None
            
    def initPerm(self):
        self.posi_perm = torch.randperm(self.posi_len)
        self.nega_perm = torch.randperm(self.nega_len)
            
    def getIndex(self,index):
        indexList = [0,0]
        indexList[0] = self.posi_perm[index%self.posi_len]
        if self.randNeg:
            indexList[1] = self.nega_perm[(index+np.random.randint(0,self.nega_len))%self.nega_len]
        else:
            indexList[1] = self.nega_perm[index%self.nega_len]
        
        return indexList

    def loadImage(self,path):
        img = PIL.Image.open(path)
        img.load()
        if self.preTransform:
            img = self.preTransform(img)

        img = tf.ToTensor()(np.array(img))
        img = img.float()/65535
        if self.nbC > 1:
            img = img.repeat(self.nbC,1,1)
        if self.normTF:
            img = self.normTF(img)
            
        if self.gn>0:
            img = img + torch.randn(img.shape)*self.gn  # @UndefinedVariable
                
        return img

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        # shuffle in this part is not necessary.

        indexList = self.getIndex(index)

        if self.pn == 1:
            po = self.loadImage(self.posi_list[indexList[0]])
            na = self.loadImage(self.nega_list[indexList[1]])
    
            # return img, target, index
            return po, na,self.posi_id[indexList[0]],self.nega_id[indexList[1]]
        elif self.pn == 2:
            po = self.loadImage(self.posi_list[indexList[0]])
            return po, self.posi_id[indexList[0]]
        elif self.pn == 3:
            na = self.loadImage(self.nega_list[indexList[1]])
            return na,self.nega_id[indexList[1]]
        elif self.pn == 4:
            po = self.loadImage(self.posi_list[indexList[0]])
            na = self.loadImage(self.nega_list[indexList[1]])
    
            # return img, target, index
            return po, na,0,1
       
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.set_len