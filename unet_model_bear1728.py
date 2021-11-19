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
# copy from unet_model_bear5.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

import unet_parts_bear1728 as upb
import util as ul



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = upb.inconv(n_channels, 64)#64x1728x1408
        self.down1 = upb.down(64, 128)#128x864x704
        self.down2 = upb.down(128, 256)#256x432x352
        
        self.down3 = upb.down(256, 512)#512x216x176
        
        self.down4 = upb.down(512, 512)#512x108x88
        
        self.down5 = upb.down(512,512)#512x54x44
        
        self.up1 = upb.up(1024, 512)#512x108x88
        
        self.up2 = upb.up(1024, 256)#256x216x176
        
        self.up3 = upb.up(512, 128)#128x432x352

        self.up4 = upb.up(256, 64)#64x864x704
        self.up5 = upb.up(128, 64)#64x1728x1408
        self.outc = upb.outconv(64, n_classes) #may need use kernel=3 instead, for future test.
        
        self.apply(ul.initialize_weights)
        

    def forward(self, x):
        x1 = self.inc(x) #64x1728x832
        x2 = self.down1(x1)#128x870x416
        x3 = self.down2(x2)#256x435x208

        x4 = self.down3(x3)#512x218x104
        x5 = self.down4(x4)#512x109x52

        x6 = self.down5(x5)#512x54x26

        x = self.up1(x6, x5)#512x108x52

        
        x = self.up2(x, x4)#256x218x104
        
        x = self.up3(x, x3)#128x436x208

        
        x = self.up4(x, x2)#64x870x416
        
        x = self.up5(x, x1)#64x1740x832
        
        x = self.outc(x)#n_classes x 1740x832
        return x
    
