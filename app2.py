"""Part I: CellQuantifier Sequence Control"""

control = [
# 'clean_dir',
# 'load',
# 'regi',
# 'mask_boundary', 'mask_53bp1', # 'mask_53bp1_blob',
# 'deno_box', 'deno_gaus',
# 'check_detect_fit',
# 'detect_fit',
# 'filt_track',
'plot_traj',
# 'phys_dist2boundary', 'phys_dist253bp1', #'phys_dist253bp1_blob',
# 'sort_plot'
]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #HEADER INFO
  'Processed By:': 'Clayton Seitz',
  'Start frame index': 0,
  'End frame index': 1,
  'Load existing analMeta': False,

  #IO
  'IO input_path': '/home/clayton/Desktop/input/',
  'IO output_path': '/home/clayton/Desktop/temp/',

  #REGISTRATION SETTINGS
  'Regi reference file label': '',
  'Regi ref_ind_num': 0,
  'Regi sig_mask': 3,
  'Regi thres_rel': 0.2,
  'Regi poly_deg': 2,
  'Regi rotation_multiplier': 1,
  'Regi translation_multiplier': 1,
  'Regi use_ransac': True,

  #SEGMENTATION SETTINGS
  'Segm min_size': 'NA',
  'Segm threshold': 'NA',

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '53bp1',
  'Mask boundary_mask sig': 3,
  'Mask boundary_mask thres_rel': 0.2,
  #MASK_53BP1 SETTINGS
  'Mask 53bp1_mask file label': '53bp1',
  'Mask 53bp1_mask sig': 3,
  'Mask 53bp1_mask thres_rel': 0.7,
  #MASK_53BP1_BLOB SETTINGS
  'Mask 53bp1_blob_mask file label': '53bp1',
  'Mask 53bp1_blob_threshold': 0.02,
  'Mask 53bp1_blob_min_sigma': 2,
  'Mask 53bp1_blob_max_sigma': 4,
  'Mask 53bp1_blob_num_sigma': 5,
  'Mask 53bp1_pk_thresh_rel': 0.15,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,
#
  #DETECTION SETTINGS
  'Det blob_threshold': 1e-6,
  'Det blob_min_sigma': 5,
  'Det blob_max_sigma': 10,
  'Det blob_num_sigma': 10,
  'Det pk_thresh_rel': 0.03,

  #TRACKING SETTINGS
  'Trak frame_rate': 3.33,
  'Trak pixel_size': 0.1084,
  'Trak divide_num': 5,
  'Trak search_range': 2,
  'Trak memory': 3,

  #FILTERING SETTINGS
  'Filt max_dist_err': 1,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 0.8,
  'Filt traj_length_thres': 50,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-20, 0],
  'Sort dist_to_53bp1': [-50, 10],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline2 import *
pipeline_batch(settings, control)
