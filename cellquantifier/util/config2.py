import pandas as pd
import os

class Config():

	def __init__(self, config):

		#I/O
		self.INPUT_PATH = config['IO input_path']
		self.OUTPUT_PATH = config['IO output_path']
		self.ROOT_NAME = ''
		self.START_FRAME = config['Start frame index']
		self.TRANGE = range(config['Start frame index'], config['End frame index'])
		self.LOAD_ANALMETA = config['Load existing analMeta']

		#REGISTRATION SETTINGS
		self.REF_FILE_NAME = ''
		self.REF_IND_NUM = config['Regi ref_ind_num']
		self.SIG_MASK = config['Regi sig_mask']
		self.THRES_REL = config['Regi thres_rel']
		self.POLY_DEG = config['Regi poly_deg']
		self.ROTATION_MULTIPLIER = config['Regi rotation_multiplier']
		self.TRANSLATION_MULTIPLIER = config['Regi translation_multiplier']
		self.USE_RANSAC = config['Regi use_ransac']

		#SEGMENTATION SETTINGS
		self.MIN_SIZE = config['Segm min_size']
		self.SEGM_THRESHOLD = config['Segm threshold']

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = config['Deno boxcar_radius']
		self.GAUS_BLUR_SIG = config['Deno gaus_blur_sig']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Det blob_threshold']
		self.MIN_SIGMA = config['Det blob_min_sigma']
		self.MAX_SIGMA = config['Det blob_max_sigma']
		self.NUM_SIGMA = config['Det blob_num_sigma']
		self.PEAK_THRESH_REL = config['Det pk_thresh_rel']

	    #TRACKING SETTINGS
		self.SEARCH_RANGE = config['Trak search_range']
		self.MEMORY = config['Trak memory']
		self.FRAME_RATE = config['Trak frame_rate']
		self.PIXEL_SIZE = config['Trak pixel_size']
		self.DIVIDE_NUM = config['Trak divide_num']

		#TRACKING FILTERING SETTINGS
		if (config['Filt max_dist_err']=='') & \
			(config['Filt max_sig_to_sigraw']=='') & \
			(config['Filt max_delta_area']=='') & \
			(config['Filt traj_length_thres']==''):
			self.DO_FILTER = False
		else:
			self.DO_FILTER = True

		self.FILTERS = {

		'MAX_DIST_ERROR': config['Filt max_dist_err'],
		'SIG_TO_SIGRAW' : config['Filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': config['Filt max_delta_area'],
		'TRAJ_LEN_THRES': config['Filt traj_length_thres']

		}

		#PHYSICS SETTINGS
		self.DIST2BOUNDARY_MASK_NAME = ''
		self.MASK_SIG_BOUNDARY = config['Mask boundary_mask sig']
		self.MASK_THRES_BOUNDARY = config['Mask boundary_mask thres_rel']
		self.DIST253BP1_MASK_NAME = ''
		self.MASK_SIG_53BP1 = config['Mask 53bp1_mask sig']
		self.MASK_THRES_53BP1 = config['Mask 53bp1_mask thres_rel']
		self.MASK_53BP1_BLOB_NAME = ''
		self.MASK_53BP1_BLOB_THRES = config['Mask 53bp1_blob_threshold']
		self.MASK_53BP1_BLOB_MINSIG = config['Mask 53bp1_blob_min_sigma']
		self.MASK_53BP1_BLOB_MAXSIG = config['Mask 53bp1_blob_max_sigma']
		self.MASK_53BP1_BLOB_NUMSIG = config['Mask 53bp1_blob_num_sigma']
		self.MASK_53BP1_BLOB_PKTHRES_REL = config['Mask 53bp1_pk_thresh_rel']

		#SORTING SETTINGS
		if (config['Sort dist_to_boundary']=='') & \
			(config['Sort dist_to_53bp1']==''):
			self.DO_SORT = False
		else:
			self.DO_SORT = True

		self.SORTERS = {

		'DIST_TO_BOUNDARY': config['Sort dist_to_boundary'],
		'DIST_TO_53BP1' : config['Sort dist_to_53bp1'],

		}

		#DICT
		self.DICT = config.copy()

	def to_posix_path(self, path):
		path = os.path.abspath(path)
		if not path.endswith('/'):
			path += '/'
		return path

	def save_config(self):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path',
									'Processed By:'])
		config_df.to_csv(path, header=False)

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))
