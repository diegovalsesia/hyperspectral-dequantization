class Config(object):
    
    def __init__(self):

        # directories
        self.data_dir = ''
        self.orig_data_file = ''
        self.quant_data_file = ''
        self.test_dir = 'test/'
        
        # input and layer params
        self.patch_size = [32, 32, 8]
        self.Q = 0
        self.mu = 1384.0
        self.std = 1507.0

        # learning
        self.batch_size = 50
        self.N_iter = 4000000
        self.learning_rate_gen = 1e-7

        # debugging
        self.save_every_iter = 10000
        self.summaries_every_iter = 50
        self.test_every_iter = 1000


    def set_data_dir(self, data_dir):
        
        self.data_dir = data_dir


    def set_data_files(self, quant_filename):
        
        self.orig_data_file = self.data_dir + "all_orig.mat"
        self.quant_data_file = self.data_dir + quant_filename        


    def set_Q(self, Q):
        
        self.Q = Q



class ConfigTest(object):
    
    def __init__(self):

        # directories
        self.data_dir = ''
        self.orig_data_file = ''
        self.quant_data_file = ''
        self.step_size = [1, 1, 1]
        self.patch_size = [512, 680, 8]
        
        # input and layer params
        self.batch_size = 1
        self.mu = 1384.0
        self.std = 1507.0


    def set_data_dir(self, data_dir):
        
        self.data_dir = data_dir


    def set_data_files(self, quant_filename):
        
        self.orig_data_file = self.data_dir + "../../../sc0_orig.mat"
        self.quant_data_file = self.data_dir + quant_filename  
