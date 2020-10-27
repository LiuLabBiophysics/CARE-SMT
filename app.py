from datetime import datetime
from cellquantifier.util.pipeline import pipeline_control
"""Part I: CellQuantifier Sequence Control"""
CF0 = 1         #load + regi
CF1 = 0        #(load) + mask
CF2 = 1         #deno
CF3 = 0         #check
CF4 = 1         #det_fit
CF5 = 0         #filt_track
CF6 = 0         #physics
CF7 = 0         #sort_plot
CF8 = 0         #merg_plot
Auto = 0        #automate smt
#                              load regi mask deno chk dt_ft fl_tk phys st_pt mg_pt video
if CF0:        Control_Flow = [  1,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0]
if CF1:        Control_Flow = [  0,   0,   1,   0,   0,   0,   0,   0,    0,   0,   0]
if CF2:        Control_Flow = [  0,   0,   0,   1,   0,   0,   0,   0,    0,   0,   0]
if CF3:        Control_Flow = [  0,   0,   0,   0,   1,   0,   0,   0,    0,   0,   0]
if CF4:        Control_Flow = [  0,   0,   0,   0,   0,   1,   0,   0,    0,   0,   1]
if CF5:        Control_Flow = [  0,   0,   0,   0,   0,   0,   1,   0,    0,   0,   0]
if CF6:        Control_Flow = [  0,   0,   0,   0,   0,   0,   0,   1,    0,   0,   0]
if CF7:        Control_Flow = [  0,   0,   0,   0,   0,   0,   0,   0,    1,   0,   0]
if CF8:        Control_Flow = [  0,   0,   0,   0,   0,   0,   0,   0,    1,   1,   0]
if Auto:       Control_Flow = [  1,   0,   1,   1,   1,   1,   1,   1,    1,   0,   1]

"""Part II: CellQuantifier Parameter Settings"""
settings = {

  #HEADER INFO
  'Processed By:': 'Clayton Seitz',
  'Date': datetime.now(),
  'Raw data file': 'simulated_cell.tif',
  'Start frame index': 0,
  'End frame index': 10,
  'Check frame index': 0,

  #IO
  'IO input_path': 'cellquantifier/data/',
  'IO output_path': '/home/cwseitz/Desktop/temp/',

  #REGISTRATION SETTINGS
  'Regi ref_ind_num': 100,
  'Regi sig_mask': 3,
  'Regi thres_rel': .1,
  'Regi poly_deg': 2,
  'Regi rotation_multplier': 1,
  'Regi translation_multiplier': 1,

  #SEGMENTATION SETTINGS
  'Segm min_size': 'NA',
  'Segm threshold': 'NA',

  #MASK SETTINGS
  'Phys dist2boundary_mask file': 'x.tif',
  'Phys dist2boundary_mask sig': 3,
  'Phys dist2boundary_mask thres_rel': 0.08,
  'Phys dist253bp1_mask file': 'x.tif',
  'Phys dist253bp1_mask sig': 1,
  'Phys dist253bp1_mask thres_rel': 0.35,

  #DENOISE SETTINGS
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,

  #DETECTION SETTINGS
  'Det blob_threshold': 0.05,
  'Det blob_min_sigma': 2,
  'Det blob_max_sigma': 4,
  'Det blob_num_sigma': 5,
  'Det pk_thresh_rel': 0.15,
  'Det plot_r': False,

  #FITTING SETTINGS
  'Fitt r_to_sigraw': 1, #determines the patch size

  #TRACKING SETTINGS
  'Trak frame_rate': 3.33,
  'Trak pixel_size': 0.1084,
  'Trak divide_num': 5,
  'Trak search_range': 2,
  'Trak memory': 3,

  #FITTING FILTERING SETTINGS
  'Filt do_filter': True,
  'Filt max_dist_err': 1,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 0.8,
  'Filt traj_length_thres': 80,

  #PHYSICS SETTINGS


  #SORTING SETTINGS
  'Sort do_sort': True,
  'Sort dist_to_boundary': [-20, 0],
  'Sort dist_to_53bp1': [-50, 10],

  #DIAGNOSTIC
  'Diag diagnostic': False,
  'Diag pltshow': False,

}

"""Part III: Run CellQuantifier"""
label = ['load', 'regi', 'mask', 'deno', 'check', 'detect_fit', 'filt_trak',
        'phys', 'sort_plot', 'merge_plot', 'video']
control = dict(zip(label, Control_Flow))
pipeline_control(settings, control)
