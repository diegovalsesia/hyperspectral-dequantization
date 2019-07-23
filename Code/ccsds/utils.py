import numpy as np
import h5py


class OrigData(object):
	def __init__(self, config, expand=False):
		super(OrigData, self).__init__()	
		orig_dataset = h5py.File(config.orig_data_file) 
		self.orig_patches = (orig_dataset['Pall'].value.astype(np.float).T - config.mu)/config.std
		if expand:
			self.orig_patches = self.orig_patches[:,:,:,:,np.newaxis]
		self.num_samp = self.orig_patches.shape[0]

	def next_batch(self, batch_size, seed=0):
		return self.sampling(batch_size, seed)

	def sampling(self, batch_size, seed):
		if seed:
			np.random.seed(seed)
		pos = np.random.choice(self.num_samp, size=batch_size)
		data_batch = self.orig_patches[pos,:,:,:]
		return data_batch


class QuantData(object):

	def __init__(self, config, expand=False):
		super(QuantData, self).__init__()	
		quant_dataset = h5py.File(config.quant_data_file) 
		self.quant_patches = (quant_dataset['PallQ'].value.astype(np.float).T - config.mu)/config.std
		if expand:
			self.quant_patches = self.quant_patches[:,:,:,:,np.newaxis]
		self.num_samp = self.quant_patches.shape[0]

	def next_batch(self, batch_size, seed=0):
		return self.sampling(batch_size, seed)

	def sampling(self, batch_size, seed):
		if seed:
			np.random.seed(seed)		
		pos = np.random.choice(self.num_samp, size=batch_size)
		data_batch = self.quant_patches[pos,:,:,:]
		return data_batch
