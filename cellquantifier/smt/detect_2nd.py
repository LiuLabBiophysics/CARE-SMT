# import numpy as np
# import pandas as pd
# import math
# from cellquantifier.qmath.gaussian_2d import (gaussian_2d, get_moments,
#                                             fit_gaussian_2d)
# from ..plot.plotutil import anno_scatter, anno_blob
# from ..plot.plotutil import plot_end
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
# from skimage.feature import blob_log
#
#
# def detect_2nd(pims_frame,
#             blobs_df,
#
#             min_sig=1,
#             max_sig=3,
#             num_sig=5,
#             blob_thres=0.1,
#             peak_thres_rel=0.1,
#             r_to_sigraw=3,
#             pixel_size = .1084,
#
#             diagnostic=True,
#             pltshow=True,
#             plot_r=True,
#             blob_marker='^',
#             blob_markersize=10,
#             blob_markercolor=(0,0,1,0.8),
#
#             truth_df=None,
#             segm_df=None,
#             ):
#     """
#     Second round of detection.
#
#     Parameters
#     ----------
#     pims_frame : pims.Frame object
#         Each frame in the format of pims.Frame.
#     bolbs_df : DataFrame
#         bolb_df with columns of 'x', 'y', 'r'.
#     diagnostic : bool, optional
#         If true, print the diagnostic strings.
#     pltshow : bool, optional
#         If true, show diagnostic plot.
#     truth_df : DataFrame, optional
#         Ground truth DataFrame with columns of 'x', 'y'.
#     segm_df : DataFrame, optional
#         Segmentation DataFrame with columns of 'x', 'y'.
#     min_sig : float, optional
# 		As 'min_sigma' argument for blob_log().
# 	max_sig : float, optional
# 		As 'max_sigma' argument for blob_log().
# 	num_sig : int, optional
# 		As 'num_sigma' argument for blob_log().
# 	blob_thres : float, optional
# 		As 'threshold' argument for blob_log().
# 	peak_thres_rel : float, optional
# 		Relative peak threshold [0,1].
# 		Blobs below this relative value are removed.
# 	r_to_sigraw : float, optional
# 		Multiplier to sigraw to decide the fitting patch radius.
# 	pixel_size : float, optional
# 		Pixel size in um. Used for the scale bar.
#
#     Returns
#     -------
#     det2nd_df : DataFrame
#         columns=['frame', 'x_raw', 'y_raw', 'sig_raw', 'r',
#                 'peak', 'mass', 'mean', 'std',
#                 'foci_num']
#     plt_array :  2d ndarray
#         2D ndarray of diagnostic plot.
#
#     Examples
#     --------
#     """
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~~Check if df_blobs is empty~~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#     if blobs_df.empty:
#         print("\n"*3)
#         print("##############################################")
#         print("ERROR: blobs_df is empty!!!")
#         print("##############################################")
#         print("\n"*3)
#         return
#
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare the dataformat~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#
#     df = pd.DataFrame([], columns=['frame', 'x', 'y', 'r', 'sig_raw',
#             'peak', 'mass', 'mean', 'std',
#             'foci_num', 'foci_df'])
#
#     df['frame'] = blobs_df['frame'].to_numpy()
#     df['x'] = blobs_df['x'].to_numpy()
#     df['y'] = blobs_df['y'].to_numpy()
#     df['r'] = blobs_df['r'].to_numpy()
#     df['sig_raw'] = blobs_df['sig_raw'].to_numpy()
#
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~Fit each blob. If fail, pass~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#     for i in df.index:
#         x0 = int(df.at[i, 'x'])
#         y0 = int(df.at[i, 'y'])
#         delta = int(round(df.at[i, 'r']))
#         patch = pims_frame[x0-delta:x0+delta+1, y0-delta:y0+delta+1]
#
#         df.at[i, 'peak'] = patch.max()
#         df.at[i, 'mass'] = patch.sum()
#         df.at[i, 'mean'] = patch.mean()
#         df.at[i, 'std'] = patch.std()
#
#         try:
#             blobs = blob_log(patch,
#             				 min_sigma=min_sig,
#             				 max_sigma=max_sig,
#             				 num_sigma=num_sig,
#             				 threshold=blob_thres)
#             foci_df = pd.DataFrame([], columns=['x', 'y', 'sig_raw', 'r', 'peak'])
#             foci_df['x'] = x0 - delta + blobs[:, 0]
#             foci_df['y'] = y0 - delta + blobs[:, 1]
#             foci_df['sig_raw'] = blobs[:, 2]
#             foci_df['r'] = blobs[:, 2] * r_to_sigraw
#
#             for j in foci_df.index:
#                 xf = int(foci_df.at[j, 'x'])
#                 yf = int(foci_df.at[j, 'y'])
#                 rf = int(round(foci_df.at[j, 'r']))
#                 foci = pims_frame[xf-rf:xf+rf+1, yf-rf:yf+rf+1]
#                 foci_df.at[j, 'peak'] = foci.max()
#
#             # """
#             # ~~~~~~~Filter detections below peak_thres_abs~~~~~~~
#             # """
#             if len(foci_df) > 1:
#                 # peak_thres_abs = (foci_df['peak'].max() - foci_df['peak'].min()) \
#                 #         * peak_thres_rel + foci_df['peak'].min()
#                 peak_thres_abs = foci_df['peak'].max() * peak_thres_rel
#                 foci_df = foci_df[ foci_df['peak'] > peak_thres_abs ]
#
#             df.at[i, 'foci_num'] = len(blobs)
#             df.at[i, 'foci_df'] = foci_df
#         except:
#             pass
#
#     det2nd_df = df
#
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#
#     plt_array = []
#     if diagnostic:
#         f1 = df.copy()
#         # """
# 	    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Show the img~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 	    # """
#         image = pims_frame
#         fig, ax = plt.subplots(figsize=(9, 9))
#         ax.imshow(image, cmap="gray", aspect='equal')
#
#         # """
#         # ~~~~~~~~~~~~~~~~~~~Annotate foci_df~~~~~~~~~~~~~~~~~~~
#         # """
#         for i in f1.index:
#             anno_blob(ax, f1.at[i, 'foci_df'],
#                     marker=blob_marker, markersize=blob_markersize,
#     				plot_r=plot_r, color=blob_markercolor)
#
#         # """
#         # ~~~~~~~~~~~~~~~~~Annotate truth_df, segm_df, det2nd_df~~~~~~~~~~~~~~~~~~~
#         # """
#         # anno_blob(ax, f1, marker='x', plot_r=1, color=(1,0,0,0.8))
#
#         if isinstance(segm_df, pd.DataFrame):
#             anno_scatter(ax, segm_df, marker='^', color=(0,0,1,0.8))
#
#         if isinstance(truth_df, pd.DataFrame):
#             anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))
#
#         plt_array = plot_end(fig, pltshow)
#
#     return det2nd_df, plt_array
#
#
# def detect_2nd_batch(pims_frames,
#             blobs_df,
#
#             min_sig=1,
#             max_sig=3,
#             num_sig=5,
#             blob_thres=0.1,
#             peak_thres_rel=0.1,
#             r_to_sigraw=3,
#             pixel_size = .1084,
#
#             diagnostic=True,
#             pltshow=True,
#             plot_r=True,
#             blob_marker='^',
#             blob_markersize=10,
#             blob_markercolor=(0,0,1,0.8),
#
#             truth_df=None,
#             segm_df=None,
#             ):
#     """
#     Point spread function fitting for the whole movie.
#
#     Parameters
#     ----------
#     See detect_2nd().
#
#     Returns
#     -------
#     det2nd_df : DataFrame
#         columns=['frame', 'x', 'y', 'sig_raw', 'r',
#                 'peak', 'mass', 'mean', 'std',
#                 'foci_num', 'foci_df']
#     plt_array :  3d ndarray
#         3D ndarray of diagnostic plot.
#
#     Examples
#     --------
#     """
#
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare the dataformat~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#
#     df = pd.DataFrame([], columns=['frame', 'x_raw', 'y_raw', 'r', 'sig_raw',
#             'peak', 'mass', 'mean', 'std',
#             'foci_num', 'foci_df'])
#     plt_array = []
#
#     # """
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Update blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # """
#     for i in range(len(pims_frames)):
#         current_frame = pims_frames[i]
#         fnum = current_frame.frame_no
#
#         current_blobs_df = blobs_df[blobs_df['frame'] == fnum]
#
#         if isinstance(truth_df, pd.DataFrame):
#             curr_truth_df = truth_df[truth_df['frame'] == fnum]
#         else:
#             curr_truth_df = None
#
#         if isinstance(segm_df, pd.DataFrame):
#             current_segm_df = segm_df[segm_df['frame'] == fnum]
#         else:
#             current_segm_df = None
#
#         tmp_det2nd_df, tmp_plt_array = detect_2nd(pims_frame=current_frame,
#                        blobs_df=current_blobs_df,
#
#                        min_sig=min_sig,
#                        max_sig=max_sig,
#                        num_sig=num_sig,
#                        blob_thres=blob_thres,
#                        peak_thres_rel=peak_thres_rel,
#                        r_to_sigraw=r_to_sigraw,
#                        pixel_size =pixel_size,
#
#                        diagnostic=diagnostic,
#                        pltshow=pltshow,
#                        plot_r=plot_r,
#                        blob_marker=blob_marker,
#                        blob_markersize=blob_markersize,
#                        blob_markercolor=blob_markercolor,
#
#                        truth_df=curr_truth_df,
#                        segm_df=current_segm_df,
#                        )
#         df = pd.concat([df, tmp_det2nd_df], sort=False)
#         plt_array.append(tmp_plt_array)
#
#     det2nd_df = df
#     plt_array = np.array(plt_array)
#
#     return det2nd_df, plt_array
