"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'mask_boundary',
# 'denoise',
# 'check_detect_fit',
# 'detect',
# 'fit',
# 'filt_track',
# 'phys_dist2boundary',
# 'phys_antigen_data',
# 'plot_traj',
# 'anim_traj',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'IO input_path': '/home/linhua/Desktop/input/',
  'IO output_path': '/home/linhua/Desktop/output/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.163,
  'Frame_rate': 2,

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '-bdr',
  'Mask boundary_mask sig': 0,
  'Mask boundary_mask thres_rel': 0.05,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.05,
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 3,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.1,

  #FILTERING SETTINGS
  'Filt max_dist_err': 3,
  'Filt max_sig_to_sigraw': 6,
  'Filt max_delta_area': 2.4,
  'Filt traj_length_thres': 20,

  #TRACKING SETTINGS
  'Trak search_range': 2.5,
  'Trak memory': 5,
  'Trak divide_num': 5,

  #SORTING SETTINGS
  'Sort dist_to_boundary': [-1000, 1000],
  'Sort travel_dist': [-1000, 1000],

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_antigen import *
pipeline_batch(settings, control)
