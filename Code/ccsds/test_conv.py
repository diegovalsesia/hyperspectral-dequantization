#!/usr/bin/env python3

from config import ConfigTest
from utils_test import QuantTestData, rebuild_image
import numpy as np

import os
import time
t0 = time.time()

import torch
import torch.autograd as autograd
from torch.autograd import Variable

import scipy.io as sio

## Parameters
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--seed', type=int)
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--G_load', default='', help='Full path to Generator model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
parser.add_argument('--Q', type=int, default=0, help='Quantization step size')
parser.add_argument('--test_dir', default='', help='Testing data directory (ex: /home/output_folder/run-5/Data/')
parser.add_argument('--test_quant_filename', default='', help='Mat file with quantized image')
parser.add_argument('--reconstruct_dir', default='', help='Directory for dequantized images')
parser.add_argument('--output_filename', default='', help='Output filename')
parser.add_argument('--sz', type=int, default=0, help='Shift step')
parser.add_argument('--verbose', default='False', help='print debug stuff')
param = parser.parse_args()

if param.cuda:
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True


## Setting seed
# import random
# if param.seed is None:
# 	param.seed = random.randint(1, 10000)
# print("Random Seed: ", param.seed)
# random.seed(param.seed)
# torch.manual_seed(param.seed)
# if param.cuda:
# 	torch.cuda.manual_seed_all(param.seed)

## Import config
config_test = ConfigTest()
config_test.set_data_dir(param.test_dir)
config_test.set_data_files(param.test_quant_filename)

## Importing dataset
quant_data = QuantTestData(config_test)

config_test.patch_size = [quant_data.img_q.shape[0], quant_data.img_q.shape[1], 8]
config_test.batch_size = 1


if param.sz==0:
	shift_step_size = config_test.step_size[-1]
else:
	shift_step_size = param.sz


## Models
class _Residual_Block(torch.nn.Module):
	def __init__(self):
		super(_Residual_Block, self).__init__()

		self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.in1 = torch.nn.InstanceNorm2d(64, affine=True)
		self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
		self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.in2 = torch.nn.InstanceNorm2d(64, affine=True)

	def forward(self, x):
		identity_data = x
		output = self.relu(self.in1(self.conv1(x)))
		output = self.in2(self.conv2(output))
		#output = self.conv2(self.relu(self.conv1(x)))
		output = self.relu(self.conv1(output))
		output = self.conv2(output)
		output = torch.add(output,identity_data)
		return output 


# network
class REC_G(torch.nn.Module):
	def __init__(self):
		super(REC_G, self).__init__()
		
		self.conv_input = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
		
		self.residual = self.make_layer(_Residual_Block, 2)

		self.conv_output = torch.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
		
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, 0.001)
				if m.bias is not None:
					m.bias.data.zero_()

	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return torch.nn.Sequential(*layers)

	def forward(self, x):
		x = x.transpose(1,3) # BHWC to BCHW conversion
		out = self.relu(self.conv_input(x))
		out = self.residual(out)
		out = self.conv_output(out)
		out = torch.clamp(out, min=-np.floor(param.Q/2)/config_test.std, max=np.floor(param.Q/2)/config_test.std)
		out = torch.add(out,x)	
		out = out.transpose(3,1) # BCHW to BHWC conversion
		return out


## Initialization
G = REC_G()

# Load existing models
if param.G_load != '':
	G.load_state_dict(torch.load(param.G_load, map_location=lambda storage, loc: storage))


# Soon to be variables
xQ = torch.FloatTensor(config_test.batch_size, config_test.patch_size[0], config_test.patch_size[1], config_test.patch_size[2])

# Everything cuda
if param.cuda:
	G = G.cuda()
	xQ = xQ.cuda()

# Now Variables
xQ = Variable(xQ)

cnt_img = np.zeros_like(quant_data.img_q)
x_hat = np.zeros_like(quant_data.img_q)
for shift_z in range(0,config_test.patch_size[2], shift_step_size):
		
	bad_data = quant_data.get_shifted_patches(config_test, 0, 0, shift_z)

	#print(bad_data.shape[0])
	#print(config_test.batch_size)
	for b in range(0,bad_data.shape[0],config_test.batch_size):

		if param.cuda:
			xQ.data = torch.from_numpy(bad_data[b:(b+config_test.batch_size),:,:,:].astype(np.float32)).cuda()
		else:
			xQ.data = torch.from_numpy(bad_data[b:(b+config_test.batch_size),:,:,:].astype(np.float32))

		x_fake = G.eval()(xQ)
		
		if b==0:
			x_dq = x_fake.cpu().data.numpy()
		else:	
			x_dq = np.concatenate( (x_dq, x_fake.cpu().data.numpy()), axis=0 )

	#print(x_dq.shape)
	#print( quant_data.img_q.shape)
	cnt_img = cnt_img + rebuild_image(np.ones_like(x_dq), config_test, 0, 0, shift_z, quant_data.img_q.shape)
	x_hat = x_hat + rebuild_image(x_dq, config_test, 0, 0, shift_z, quant_data.img_q.shape)
	
	if param.verbose != "False":
		print('+x: %d, +y: %d, +z: %d' % (0,0,shift_z))

# average
x_hat = (x_hat / cnt_img.astype(np.double))*config_test.std + config_test.mu;

sio.savemat('%s/%s' % (param.reconstruct_dir, param.output_filename),{'x_hat':x_hat})
