"""Part I: CellQuantifier Sequence Control"""

control = [
# 'clean_dir',
# 'load',
# 'regi',
# 'mask_boundary',
# 'mask_53bp1',
# 'mask_53bp1_blob',
# 'deno_mean',
# 'deno_median',
# 'deno_box',
# 'deno_gaus',
# 'check_detect_fit',
# 'detect',
# 'fit',

# 'filt_track',
# 'phys_xy_global',
# 'phys_dist2halfcilia',
# 'phys_cilia_halfsign',
# 'phys_cilia_otherinfo',
# 'plot_traj',
'anim_traj',
# 'anno_kymo',
# 'merge_plot',

# 'phys_dist253bp1',
# 'phys_dist253bp1_blob',
# 'sort_plot',
# 'merge_plot',

# 'filt_track',
# 'phys_dist2boundary',
# 'phys_antigen_data',
# 'plot_traj',
# 'anim_traj',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {
    #DETECTION SETTINGS
    'Det blob_threshold': 'auto',
    'Det blob_min_sigma': 2,
    'Det blob_max_sigma': 3,
    'Det blob_num_sigma': 50,
    'Det pk_thresh_rel': 'auto',
    'Det mean_thresh_rel': 0,
    'Det r_to_sigraw': 1,

  #IO
  'IO input_path': '/home/linhua/Desktop/input/',
  'IO output_path': '/home/linhua/Desktop/input/',

  #HEADER INFO
  'Processed By:': 'Hua Lin',
  'Start frame index': 0,
  'End frame index': 600,
  'Load existing analMeta': False,

  #REGISTRATION SETTINGS
  'Regi reference file label': '',
  'Regi ref_ind_num': '',
  'Regi sig_mask': '',
  'Regi thres_rel': '',
  'Regi poly_deg': '',
  'Regi rotation_multiplier': '',
  'Regi translation_multiplier': '',
  'Regi use_ransac': '',

  #SEGMENTATION SETTINGS
  'Segm min_size': '',
  'Segm threshold': '',

  #MASK_BOUNDARY SETTINGS
  'Mask boundary_mask file label': '',
  'Mask boundary_mask sig': '',
  'Mask boundary_mask thres_rel': '',
  #MASK_53BP1 SETTINGS
  'Mask 53bp1_mask file label': '',
  'Mask 53bp1_mask sig': '',
  'Mask 53bp1_mask thres_rel': '',
  #MASK_53BP1_BLOB SETTINGS
  'Mask 53bp1_blob_mask file label': '',
  'Mask 53bp1_blob_threshold': '',
  'Mask 53bp1_blob_min_sigma': '',
  'Mask 53bp1_blob_max_sigma': '',
  'Mask 53bp1_blob_num_sigma': '',
  'Mask 53bp1_blob_pk_thresh_rel': '',
  'Mask 53bp1_blob_search_range': '',
  'Mask 53bp1_blob_memory': '',
  'Mask 53bp1_blob_traj_length_thres': '',

  #DENOISE SETTINGS
  'Deno mean_radius': '',
  'Deno median_radius': '',
  'Deno boxcar_radius': 10,
  'Deno gaus_blur_sig': 0.5,
  'Deno minimum_radius': 7,



  #DETECTION_2ND SETTINGS
  'Det2nd blob_threshold': '',
  'Det2nd blob_min_sigma': '',
  'Det2nd blob_max_sigma': '',
  'Det2nd blob_num_sigma': '',
  'Det2nd pk_thresh_rel': '',
  'Det2nd r_to_sigraw': '',

  #TRACKING SETTINGS
  'Trak frame_rate': 1,
  'Trak pixel_size': 0.108,
  'Trak divide_num': 1,

  ###############################################
  'Trak search_range': 7,  # NO. 1
  ###############################################

  'Trak memory': 3,

  #FILTERING SETTINGS
  'Filt max_dist_err': 1,
  'Filt max_sig_to_sigraw': 2,
  'Filt max_delta_area': 1,

  ###############################################
  'Filt traj_length_thres': 15, # NO. 2
  #SORTING SETTINGS
  'Sort dist_to_boundary': '', # NO. 3
  'Sort travel_dist': '', # NO. 4
  ###############################################

  'Sort dist_to_53bp1': '',

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline3_cilia2 import *
pipeline_batch(settings, control)

# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/temp/200303_50NcBLM-physDataMerged.csv',
#                 index_col=False)
# df= df[ ~df['raw_data'].isin(['200211_50NcLiving_D1-HT-physData.csv',
#                         '200211_50NcLiving_A2-HT-physData.csv',
#                         '200206_50NcLiving_L2-HT-physData.csv',
#                         '190925_50NcLiving_K1-HT-physData.csv']) ]
#
# # df= df[ df['raw_data'].isin(['190924_50NcLiving_B1-HT-physData.csv',
# #                         '190924_50NcLiving_D1-HT-physData.csv',
# #                         '190924_50NcLiving_E1-HT-physData.csv',
# #                         '190924_50NcLiving_I1-HT-physData.csv',
# #                         '190924_50NcLiving_J2-HT-physData.csv',
# #                         '191010_50NcLiving_B1-HT-physData.csv',
# #
# #                         '191004_50NcBLM_A1-HT-physData.csv',
# #                         '191004_50NcBLM_B1-HT-physData.csv',
# #                         '191004_50NcBLM_C1-HT-physData.csv',
# #                         '191004_50NcBLM_E1-HT-physData.csv',
# #                         '191004_50NcBLM_F1-HT-physData.csv',
# #                         '191004_50NcBLM_I1-HT-physData.csv',
# #                         '191004_50NcBLM_K1-HT-physData.csv',
# #                         '191004_50NcBLM_Q1-HT-physData.csv',
# #                         '191004_50NcBLM_T1-HT-physData.csv'
# #                         ]) ]
#
# df['date'] = df['raw_data'].astype(str).str[0:6]
# df = df[ df['date'].isin(['191004', '190924', '190925', '191010', '200206',]) ]
# print(len(df))
# fig_quick_nucleosome(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_merge import *
# df = pd.read_csv('/home/linhua/Desktop/josh/200730_50NcFixed-physDataMerged.csv',
#                 index_col=False)
# fig_quick_merge(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_merge2 import *
# df = pd.read_csv('/home/linhua/Desktop/BMT/200902_50NcLiving-physDataMerged.csv',
#                 index_col=False)
# fig_quick_merge(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_merge3 import *
# df = pd.read_csv('/home/linhua/Desktop/BMT/200810_50NcLivingBMT-physDataMerged.csv',
#                 index_col=False)
# fig_quick_merge(df)


# from cellquantifier.publish import *
# import pandas as pd
# df = pd.read_csv('/home/linhua/Desktop/output/cilia-physData.csv',
#                 index_col=False)
# fig_quick_msd(df)

# import pandas as pd
# from cellquantifier.publish._fig_quick_cilia_5 import *
# from cellquantifier.phys import *
# df = pd.read_csv('/home/linhua/Desktop/phys/200619-physDataMerged.csv',
#                 index_col=False)
# df = add_travel_dist(df)
# df = add_speed(df)
# df.to_csv('/home/linhua/Desktop/phys/200619-physDataMerged2.csv',
#                 index=False)
# fig_quick_cilia_5(df)


# from skimage.io import imread, imsave
# from cellquantifier.publish import *
# from cellquantifier.video import *
# import pandas as pd
#
# df = pd.read_csv('/home/linhua/Desktop/temp/200205_MalKN-E-physData.csv',
#                 index_col=False)
# df = df[ df['traj_length']>20 ]
# fig_quick_antigen(df)


# df = pd.read_csv('/home/linhua/Desktop/temp-E/200205_MalKN-E-physData.csv',
#                 index_col=False)
# tif = imread('/home/linhua/Desktop/temp-E/200205_MalKN-E-raw.tif')[0:50]
# df = df[ (df['traj_length']>100) & (df['frame']<50) ]
# anim_tif = anim_traj(df, tif,
#                 pixel_size=0.163,
#                 cb_min=2000, cb_max=10000,
#                 cb_major_ticker=2000, cb_minor_ticker=2000,
#                 show_image=True)
# anim_tif = anim_blob(df, tif, pixel_size=0.163,
#                 show_image=False)
# imsave('/home/linhua/Desktop/temp-E/anim-traj-result.tif', anim_tif)
# fig_quick_antigen(df)


# from cellquantifier.publish import *
# import pandas as pd
#
# df = pd.read_csv('/home/linhua/Desktop/temp/200426_physDataMerged.csv',
#                 index_col=False)
# fig_quick_antigen_3(df)
