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
import torch.nn as nn
import torch.nn.functional as F
import util  # @UnresolvedImport




class ResDownBlock(nn.Module):
	
	def __init__(self,inplanes,outplanes,stride=1,downsample=None,bnmode=0):
		super(ResDownBlock,self).__init__()
		self.conv1 = nn.Conv2d(
			inplanes,outplanes,kernel_size=1,stride=1,bias=True)
		self.bn1 = util.bn2d(outplanes,mode=bnmode)
		self.conv2 = util.conv3x3(outplanes, outplanes, stride=stride)
		self.bn2 = util.bn2d(outplanes,mode=bnmode)
		self.conv3 = nn.Conv2d(outplanes,outplanes,kernel_size=1,
			stride=1,bias=True)
		self.bn3 = util.bn2d(outplanes,mode=bnmode)

		self.downsample = downsample;
		self.stride = stride
		self.bnmode = bnmode
		

	def forward(self,x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = F.leaky_relu(out, inplace=True)

		out = self.conv2(out)
		out = self.bn2(out)
		out = F.leaky_relu(out, inplace=True)

		out = self.conv3(out)
		out = self.bn3(out)


		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = F.leaky_relu(out,inplace=True)

		return out

class ResUpBlock(nn.Module):

	def __init__(self,inplanes,outplanes,stride=2,upsample=None,bnmode=0):
		super(ResUpBlock,self).__init__()
		self.conv1 = nn.Conv2d(inplanes,outplanes,kernel_size=1,
			stride=1,bias=True)
		self.bn1 = util.bn2d(outplanes,mode=bnmode)
		self.dconv2 = util.dconv3x3(outplanes,outplanes,stride=stride)
		self.bn2 = util.bn2d(outplanes,mode=bnmode)
		self.conv3 = nn.Conv2d(outplanes,outplanes,kernel_size=1,
			stride=1,bias=True)
		self.bn3 = util.bn2d(outplanes,mode=bnmode)

		self.upsample = upsample
		self.stride = stride

	def forward(self,x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = F.leaky_relu(out,inplace=True)

		out = self.dconv2(out)
		out = self.bn2(out)
		out = F.leaky_relu(out,inplace=True)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.upsample is not None:
			residual = self.upsample(x)

		out += residual
		out = F.leaky_relu(out,inplace=True)

		return out


