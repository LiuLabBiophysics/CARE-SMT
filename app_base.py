"""Part I: CellQuantifier Sequence Control"""

control = [
# 'load',
# 'deno_box',
# 'deno_gaus',
# 'check_detect_fit',
# 'detect',
# 'fit',
# 'filt_track',
# 'plot_traj',
# 'anim_traj',
# 'merge_plot',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #GENERAL INFO
  'IO input_path': '/home/linhua/Desktop/BMT/',
  'IO output_path': '/home/linhua/Desktop/BMT/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.108,
  'Frame_rate': 1,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 'auto',
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 3,
  'Det blob_num_sigma': 50,
  'Det pk_thresh_rel': 'auto',

  #TRACKING SETTINGS
  'Trak divide_num': 1,
  'Trak search_range': 7,
  'Trak memory': 3,

  #FILTERING SETTINGS
  'Filt max_dist_err': 1,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 1,
  'Filt traj_length_thres': 3,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_base.py import *
pipeline_batch(settings, control)
