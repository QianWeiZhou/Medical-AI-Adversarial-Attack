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
copy from imageDataset4f3328, for classification task.
'''
import torch
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as tf
import PIL.Image
import re


class ImageDataset(data.Dataset):  # 需要继承data.Dataset
    def __init__(self, transform=None, setType=0, rootFolder='../data/',
                 nbChannel=3, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
        # TODO
        # 1. Initialize file path or list of file names.
        self.rootFolder = rootFolder;
        self.nbC = nbChannel
        self.imean = img_mean
        self.istd = img_std
        self.pn = setType
        self.negative = ['%s/negb9List-train.txt' % self.rootFolder,
                         '%s/negb9List-valid.txt' % self.rootFolder,
                         '%s/negb9List-test.txt' % self.rootFolder, ]
        self.positive = ['%s/cancerList-train.txt' % self.rootFolder,
                         '%s/cancerList-valid.txt' % self.rootFolder,
                         '%s/cancerList-test.txt' % self.rootFolder]

        self.negative = self.negative[self.pn]
        self.positive = self.positive[self.pn]
        print(self.negative, self.positive)

        self.regex = "\d+"
        self.posi_list = []
        self.posi_id = []
        self.nega_list = []
        self.nega_id = []

        file = open(self.positive, 'r', encoding='utf-8')
        line = file.readline()
        while line:
            line = line.strip('\n').strip()
            if line[0] != '#':
                path, filename = os.path.split(line)
                temp_id = int(re.search(self.regex, filename).group())

                fh = '%s/%s' % (self.rootFolder, line)

                if fh != None:
                    self.posi_id.append(temp_id)
                    self.posi_list.append(fh)
            else:
                print('pass==> {}'.format(line))

            line = file.readline()

        file.close()
 
        file = open(self.negative, 'r', encoding='utf-8')
        line = file.readline()
        while line:
            line = line.strip('\n').strip()
            if line[0] != '#':
                path, filename = os.path.split(line)
                temp_id = int(re.search(self.regex, filename).group())

                fh = '%s/%s' % (self.rootFolder, line)

                if fh != None:
                    self.nega_id.append(temp_id)
                    self.nega_list.append(fh)

            else:
                print('pass==> {}'.format(line))

            line = file.readline()

        file.close()

        self.posi_len = self.posi_list.__len__()
        self.nega_len = self.nega_list.__len__()

        self.maxLen = np.max([self.posi_len, self.nega_len])
        self.set_len = self.maxLen * 2  # @UndefinedVariable
        print(self.posi_len, self.nega_len, self.set_len)

        self.preTransform = transform
        self.initPerm()

        if self.imean:
            self.normTF = tf.Normalize(self.imean, self.istd)
        else:
            self.normTF = None

    def initPerm(self):
        self.posi_perm = torch.randperm(self.posi_len)
        self.nega_perm = torch.randperm(self.nega_len)

    def getIndex(self, index):

        Target = 0
        if index >= self.maxLen:
            Target = 1
            index = self.nega_perm[(index - self.maxLen) % self.nega_len]
        else:
            index = self.posi_perm[index % self.posi_len]

        return index, Target

    def loadImage(self, path):
        img = PIL.Image.open(path)
        img.load()
        if self.preTransform:
            img = self.preTransform(img)
        img = tf.ToTensor()(np.array(img))
        img = img.float() / 65535
        if self.nbC > 1:
            img = img.repeat(self.nbC, 1, 1)
        if self.normTF:
            img = self.normTF(img)
        return img

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        # shuffle in this part is not necessary.

        index, Target = self.getIndex(index)

        img = None
        img_id = None
        if Target == 0:
            img = self.loadImage(self.posi_list[index])
            img_id = self.posi_id[index]
        else:
            img = self.loadImage(self.nega_list[index])
            img_id = self.nega_id[index]

        return img, Target, img_id

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.set_len
