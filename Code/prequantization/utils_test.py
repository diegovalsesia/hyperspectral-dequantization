import numpy as np
import h5py


def rebuild_image(patches, config, shift_x, shift_y, shift_z, img_shape):
	
	img_Nrows = img_shape[0]
	img_Ncols = img_shape[1]
	img_Nbands = img_shape[2]

	x_point = np.arange(0+shift_x, img_Ncols, config.patch_size[1]);
	y_point = np.arange(0+shift_y, img_Nrows, config.patch_size[0]);
	z_point = np.arange(0+shift_z, img_Nbands, config.patch_size[2]);

	if x_point[-1]+config.patch_size[1] > img_Ncols:
		x_point = x_point[:-1]
	if y_point[-1]+config.patch_size[0] > img_Nrows:
		y_point = y_point[:-1]
	if z_point[-1]+config.patch_size[2] > img_Nbands:
		z_point = z_point[:-1]

	x_hat = np.zeros(img_shape)
	iter_no=0;
	for xx in x_point:
		for yy in y_point:
			for zz in z_point:
				x_hat[yy:yy+config.patch_size[0], xx:xx+config.patch_size[1], zz:zz+config.patch_size[2]] = patches[iter_no,:,:,:];
				iter_no = iter_no+1;

	return x_hat



class QuantTestData(object):

	def __init__(self, config, expand=False):
		super(QuantTestData, self).__init__()	
		
		quant_dataset = h5py.File(config.quant_data_file) 
		self.img_q = (quant_dataset['img_q'].value.astype(np.float).T - config.mu)/config.std


	def get_shifted_patches(self, config, shift_x, shift_y, shift_z):

		img_Nrows, img_Ncols, img_Nbands = self.img_q.shape

		x_point = np.arange(0+shift_x, img_Ncols, config.patch_size[1]);
		y_point = np.arange(0+shift_y, img_Nrows, config.patch_size[0]);
		z_point = np.arange(0+shift_z, img_Nbands, config.patch_size[2]);

		if x_point[-1]+config.patch_size[1] > img_Ncols:
			x_point = x_point[:-1]
		if y_point[-1]+config.patch_size[0] > img_Nrows:
			y_point = y_point[:-1]
		if z_point[-1]+config.patch_size[2] > img_Nbands:
			z_point = z_point[:-1]

		Npatches = len(x_point)*len(y_point)*len(z_point);

		iter_no = 0;
		PallQ = np.zeros((Npatches, config.patch_size[0], config.patch_size[1], config.patch_size[2]));
		for xx in x_point:
			for yy in y_point:
				for zz in z_point:
					PallQ[iter_no,:,:,:] = self.img_q[yy:yy+config.patch_size[0], xx:xx+config.patch_size[1], zz:zz+config.patch_size[2]];
					iter_no = iter_no+1;	

		return PallQ	  



class OrigTestData(object):

	def __init__(self, config, expand=False):
		super(OrigTestData, self).__init__()	
		
		quant_dataset = h5py.File(config.orig_data_file) 
		self.x = (quant_dataset['x'].value.astype(np.float).T - config.mu)/config.std


	def get_shifted_patches(self, config, shift_x, shift_y, shift_z):

		img_Nrows, img_Ncols, img_Nbands = self.x.shape

		x_point = np.arange(0+shift_x, img_Ncols, config.patch_size[1]);
		y_point = np.arange(0+shift_y, img_Nrows, config.patch_size[0]);
		z_point = np.arange(0+shift_z, img_Nbands, config.patch_size[2]);

		if x_point[-1]+config.patch_size[1] > img_Ncols:
			x_point = x_point[:-1]
		if y_point[-1]+config.patch_size[0] > img_Nrows:
			y_point = y_point[:-1]
		if z_point[-1]+config.patch_size[2] > img_Nbands:
			z_point = z_point[:-1]

		Npatches = len(x_point)*len(y_point)*len(z_point);

		iter_no = 0;
		PallQ = np.zeros((Npatches, config.patch_size[0], config.patch_size[1], config.patch_size[2]));
		for xx in x_point:
			for yy in y_point:
				for zz in z_point:
					PallQ[iter_no,:,:,:] = self.x[yy:yy+config.patch_size[0], xx:xx+config.patch_size[1], zz:zz+config.patch_size[2]];
					iter_no = iter_no+1;	

		return PallQ	          