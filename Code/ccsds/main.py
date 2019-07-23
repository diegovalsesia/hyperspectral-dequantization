#!/usr/bin/env python3

from config import Config, ConfigTest
from utils import OrigData, QuantData
from utils_test import OrigTestData, QuantTestData, rebuild_image
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
parser.add_argument('--G_load', default='', help='Full path to Generator model to load (ex: /home/output_folder/run-5/models/G_epoch_11.pth)')
parser.add_argument('--start_iter', type=int, default=0, help='Start iteration (ex: 10001)')
parser.add_argument('--save_dir', default='', help='Save directory (ex: /home/output_folder/run-5/models/')
parser.add_argument('--log_dir', default='', help='Log directory (ex: /home/output_folder/run-5/log_dir/')
parser.add_argument('--data_dir', default='', help='Training data directory (ex: /home/output_folder/run-5/Data/')
parser.add_argument('--quant_filename', default='', help='Mat file with quantized patches')
parser.add_argument('--Q', type=int, default=0, help='Quantization step size')
parser.add_argument('--test_dir', default='', help='Testing data directory (ex: /home/output_folder/run-5/Data/')
parser.add_argument('--test_quant_filename', default='', help='Mat file with quantized patches')
param = parser.parse_args()

if param.cuda:
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True



## Setting seed
#import random
#if param.seed is None:
#	param.seed = random.randint(1, 10000)
#print("Random Seed: ", param.seed)
#random.seed(param.seed)
#torch.manual_seed(param.seed)
#if param.cuda:
#	torch.cuda.manual_seed_all(param.seed)

## Import config
config = Config()
config.set_data_dir(param.data_dir)
config.set_data_files(param.quant_filename)
config.set_Q(param.Q)

log_output = open(param.log_dir+"log.txt", 'w')
# For plotting the Loss of D and G using tensorboard
from tensorboard_logger import configure, log_value
configure(param.log_dir, flush_secs=5)

## Importing dataset
orig_data = OrigData(config)
quant_data = QuantData(config)

#test
config_test = ConfigTest()
config_test.set_data_dir(param.test_dir)
config_test.set_data_files(param.test_quant_filename)
orig_data_test = OrigTestData(config_test)
quant_data_test = QuantTestData(config_test)
bad_data_test = quant_data_test.get_shifted_patches(config_test, 0, 0, 0)

# Set a seed
np.random.seed(47)


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
		out = torch.clamp(out, min=-np.floor(config.Q/2)/config.std, max=np.floor(config.Q/2)/config.std)
		out = torch.add(out,x)	
		out = out.transpose(3,1) # BCHW to BHWC conversion
		return out


## Initialization
G = REC_G()
#G.apply(init_weights)

# Load existing models
if param.G_load != '' and param.start_iter > 1:
	G.load_state_dict(torch.load(param.G_load))
if param.G_load == '' and param.start_iter > 1:
	G.load_state_dict(torch.load('%sG_last.pth' % (param.save_dir)))

# Soon to be variables
x = torch.FloatTensor(config.batch_size, config.patch_size[0], config.patch_size[1], config.patch_size[2])
xQ = torch.FloatTensor(config.batch_size, config.patch_size[0], config.patch_size[1], config.patch_size[2])

# Everything cuda
if param.cuda:
	G = G.cuda()
	x = x.cuda()
	xQ = xQ.cuda()

# Now Variables
x = Variable(x)
xQ = Variable(xQ)

# Optimizer
optimizerG = torch.optim.Adam(G.parameters(), lr=config.learning_rate_gen, betas=(0.5, 0.9))

for p in G.parameters():
	p.requires_grad = True

## Fitting model
for i in range(param.start_iter, config.N_iter):

	# Sample quantized data
	s = np.random.randint(1,2**32-1)
	bad_data = quant_data.next_batch(config.batch_size, seed=s)
	real_data = orig_data.next_batch(config.batch_size, seed=s)
	if param.cuda:
		xQ.data = torch.from_numpy(bad_data.astype(np.float32)).cuda()
		x.data = torch.from_numpy(real_data.astype(np.float32)).cuda()
	else:
		xQ.data = torch.from_numpy(bad_data.astype(np.float32))
		x.data = torch.from_numpy(real_data.astype(np.float32))

	x_fake = G(xQ)
	
	#mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
	#errG = mse_loss(x_fake,x)
	errG = torch.mean(torch.sum( torch.sum(torch.sum(torch.mul(x_fake-x,x_fake-x), dim=1), dim=1), dim=1) )
	errG.backward()
	optimizerG.step()
	
	# Tensorboard
	if i % config.summaries_every_iter == 0:
		log_value('errG', errG.data[0], i)

	# Loss info
	#if i % 50 == 0:
	#	print('[i=%d, t=%.2f] Loss_G: %.4f' % (i, time.time()-t0, errG.data[0]))
	#	print('[i=%d, t=%.2f] Loss_G: %.4f' % (i, time.time()-t0, errG.data[0]), file=log_output)
	#	sio.savemat('%sdebug.mat' % config.debug_dir,{'x': x.cpu().data.numpy(), 'x_fake':x_fake.cpu().data.numpy(), 'xQ': xQ.cpu().data.numpy()})
	
	# Save models
	if i % config.save_every_iter == 0:
		torch.save(G.state_dict(), '%sG_%d.pth' % (param.save_dir, i))
		torch.save(G.state_dict(), '%sG_last.pth' % (param.save_dir))

	# Test info
	if i % config.test_every_iter == 0:
		x_hat = np.zeros_like(quant_data_test.img_q)
		for shift_z in range(0,config_test.patch_size[2], config_test.patch_size[2]):
			
			for b in range(0,bad_data_test.shape[0],config_test.batch_size):

				if param.cuda:
					xQ.data = torch.from_numpy(bad_data_test[b:(b+config_test.batch_size),:,:,:].astype(np.float32)).cuda()
				else:
					xQ.data = torch.from_numpy(bad_data_test[b:(b+config_test.batch_size),:,:,:].astype(np.float32))

				x_fake = G.eval()(xQ)
				
				if b==0:
					x_dq = x_fake.cpu().data.numpy()
				else:	
					x_dq = np.concatenate( (x_dq, x_fake.cpu().data.numpy()), axis=0 )

			x_hat = x_hat + rebuild_image(x_dq, config_test, 0, 0, shift_z, quant_data_test.img_q.shape)

			SNR_test = 10*np.log10(np.sum((orig_data_test.x)**2)/np.sum((x_hat-orig_data_test.x)**2))
			log_value('SNR_test', SNR_test, i)

	# for launcher
	if i % config.save_every_iter == 0:
		with open(param.log_dir+'start_iter', "w") as text_file:
			text_file.write("%d" % i)