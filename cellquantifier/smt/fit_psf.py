import numpy as np
import pandas as pd
import math
from cellquantifier.qmath.gaussian_2d import (gaussian_2d, get_moments,
                                            fit_gaussian_2d)
from ..plot.plotutil import anno_scatter, anno_blob
from ..plot.plotutil import plot_end
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def fit_psf(pims_frame,
            blobs_df,
            diagnostic=True,
            pltshow=True,
            diag_max_dist_err=1,
            diag_max_sig_to_sigraw = 3,
            truth_df=None,
            segm_df=None):
    """
    Point spread function fitting for each frame.

    Parameters
    ----------
    pims_frame : pims.Frame object
        Each frame in the format of pims.Frame.
    bolbs_df : DataFrame
        bolb_df with columns of 'x', 'y', 'r'.
    diagnostic : bool, optional
        If true, print the diagnostic strings.
    pltshow : bool, optional
        If true, show diagnostic plot.
    diag_max_dist_err : float, optional
        Virtual max_dist_err filter.
    diag_max_sig_to_sigraw : float, optional
        Virtual diag_max_sig_to_sigraw filter.
    truth_df : DataFrame, optional
        Ground truth DataFrame with columns of 'x', 'y'.
    segm_df : DataFrame, optional
        Segmentation DataFrame with columns of 'x', 'y'.

    Returns
    -------
    psf_df : DataFrame
        columns=['frame', 'x_raw', 'y_raw', 'sig_raw', 'r',
                'peak', 'mass', 'mean', 'std',
                'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
                'area', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw', dist_to_com'
                'ring_label']
    plt_array :  2d ndarray
        2D ndarray of diagnostic plot.

    Examples
    --------
    import pims
    from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
    from cellquantifier.smt.fit_psf import fit_psf, fit_psf_batch
    frames = pims.open('cellquantifier/data/simulated_cell.tif')
    blobs_df, det_plt_array = detect_blobs(frames[0], diagnostic=0)
    psf_df, fit_plt_array = fit_psf(frames[0], blobs_df)
    """
    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~Check if df_blobs is empty~~~~~~~~~~~~~~~~~~~~~~~~
    # """
    if blobs_df.empty:
        print("\n"*3)
        print("##############################################")
        print("ERROR: blobs_df is empty!!!")
        print("##############################################")
        print("\n"*3)
        # return

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare the dataformat~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    df = pd.DataFrame([], columns=['frame', 'x_raw', 'y_raw', 'r', 'sig_raw',
            'peak', 'mass', 'mean', 'std',
            'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
            'area', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw'])

    df['frame'] = blobs_df['frame'].to_numpy()
    df['x_raw'] = blobs_df['x'].to_numpy()
    df['y_raw'] = blobs_df['y'].to_numpy()
    df['r'] = blobs_df['r'].to_numpy()
    df['sig_raw'] = blobs_df['sig_raw'].to_numpy()

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~Fit each blob. If fail, pass~~~~~~~~~~~~~~~~~~~~~~~
    # """

    good_fitting_num = 0
    for i in df.index:
        x0 = int(df.at[i, 'x_raw'])
        y0 = int(df.at[i, 'y_raw'])
        delta = int(round(df.at[i, 'r']))
        patch = pims_frame[x0-delta:x0+delta+1, y0-delta:y0+delta+1]

        df.at[i, 'peak'] = patch.max()
        df.at[i, 'mass'] = patch.sum()
        df.at[i, 'mean'] = patch.mean()
        df.at[i, 'std'] = patch.std()

        try:
            p, p_err = fit_gaussian_2d(patch)
            A = p[0]
            x0_refined = x0 - delta + p[1]
            y0_refined = y0 - delta + p[2]
            sig_x = p[3]
            sig_y = p[4]
            phi = p[5]
            sig_raw = df.at[i, 'sig_raw']
            df.at[i, 'A'] = A
            df.at[i, 'x'] = x0_refined
            df.at[i, 'y'] = y0_refined
            df.at[i, 'sig_x'] = sig_x
            df.at[i, 'sig_y'] = sig_y
            df.at[i, 'phi'] = phi
            df.at[i, 'area'] = np.pi * sig_x * sig_y
            # df.at[i, 'mass'] = patch.sum()
            df.at[i, 'dist_err'] = ((x0_refined - x0)**2 + \
                            (y0_refined - y0)**2) ** 0.5
            df.at[i, 'sigx_to_sigraw'] = sig_x / sig_raw
            df.at[i, 'sigy_to_sigraw'] = sig_y / sig_raw

            # """
            # ~~~~~~~~Count the good fitting number with virtual filters~~~~~~~~
            # """

            if (x0_refined - x0)**2 + (y0_refined - y0)**2 \
                    < (diag_max_dist_err)**2 \
            and sig_x < sig_raw * diag_max_sig_to_sigraw \
            and sig_y < sig_raw * diag_max_sig_to_sigraw :
                good_fitting_num = good_fitting_num + 1
        except:
            pass

    try:
        print("Predict good fitting number and ratio in frame %d: [%d, %.2f]" %
            (pims_frame.frame_no, good_fitting_num,
            good_fitting_num/len(blobs_df)))
    except:
        pass
        
    psf_df = df

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    plt_array = []
    if diagnostic:
        # """
	    # ~~~~~~~~~~~~~~~~~~~~~Print foci num after filters~~~~~~~~~~~~~~~~~~~~~
	    # """
        f1 = df.copy()
        df_filt = pd.DataFrame([], columns=['tot_foci_num'],
                index=['detected', 'fit_success', 'dist_err', 'sigx_to_sigraw',
                        'sigy_to_sigraw'])
        df_filt.loc['detected'] = len(f1)
        f1 = f1.dropna(how='any', subset=['x', 'y'])
        df_filt.loc['fit_success'] = len(f1)
        f1 = f1[ f1['dist_err']<diag_max_dist_err ]
        df_filt.loc['dist_err'] = len(f1)
        f1 = f1[ f1['sigx_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigx_to_sigraw'] = len(f1)
        f1 = f1[ f1['sigy_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigy_to_sigraw'] = len(f1)

        # """
	    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Show the img~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    # """
        image = pims_frame
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(image, cmap="gray")

        # """
        # ~~~~~~~~~~~~~~~~~~~Add fitting contour to the image~~~~~~~~~~~~~~~~~~~
        # """
        for i in f1.index:
            Fitting_X = np.indices(image.shape)
            p0,p1,p2,p3,p4,p5 = (f1.at[i,'A'], f1.at[i,'x'], f1.at[i,'y'],
                    f1.at[i,'sig_x'], f1.at[i,'sig_y'], f1.at[i,'phi'])
            Fitting_img = gaussian_2d(Fitting_X,p0,p1,p2,p3,p4,p5)
            contour_img = np.zeros(image.shape)
            x1,y1,r1 = f1.at[i,'x'], f1.at[i,'y'], f1.at[i,'r']
            x1 = int(round(x1))
            y1 = int(round(y1))
            r1 = int(round(r1))
            contour_img[x1-r1:x1+r1+1,
                        y1-r1:y1+r1+1] = Fitting_img[x1-r1:x1+r1+1,
                                                     y1-r1:y1+r1+1]
            ax.contour(contour_img, cmap='cool')

        # """
        # ~~~~~~~~~~~~~~~~~Annotate truth_df, segm_df, psf_df~~~~~~~~~~~~~~~~~~~
        # """
        anno_blob(ax, f1, marker='x', plot_r=1, color=(1,0,0,0.8))

        if isinstance(segm_df, pd.DataFrame):
            anno_scatter(ax, segm_df, marker='^', color=(0,0,1,0.8))

        if isinstance(truth_df, pd.DataFrame):
            anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))

        # """
        # ~~~~~~~~~~~~~~~~Print predict good fitting foci num~~~~~~~~~~~~~~~~~~
        # """
        ax.text(0.95,
                0.00,
                """
                Predict good fitting foci num and ratio: %d, %.2f
                """ %(good_fitting_num, good_fitting_num/len(blobs_df)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (1, 1, 1, 0.8),
                transform=ax.transAxes)
        plt_array = plot_end(fig, pltshow)

    return psf_df, plt_array


def fit_psf_batch(pims_frames,
            blobs_df,
            diagnostic=False,
            pltshow=False,
            diag_max_dist_err=1,
            diag_max_sig_to_sigraw = 3,
            truth_df=None,
            segm_df=None):
    """
    Point spread function fitting for the whole movie.

    Parameters
    ----------
    See fit_psf().

    Returns
    -------
    psf_df : DataFrame
        columns=['frame', 'x_raw', 'y_raw', 'r',
                'A', 'x', 'y', 'sig_x', 'sig_y', 'phi','area', 'mass',
                'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw']
    plt_array :  3d ndarray
        3D ndarray of diagnostic plot.

    Examples
    --------
    import pims
    from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
    from cellquantifier.smt.fit_psf import fit_psf, fit_psf_batch
    frames = pims.open('cellquantifier/data/simulated_cell.tif')
    blobs_df, det_plt_array = detect_blobs_batch(frames)
    psf_df, fit_plt_array = fit_psf_batch(frames, blobs_df)
    """

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare the dataformat~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    df = pd.DataFrame([], columns=['frame', 'x_raw', 'y_raw', 'r', 'sig_raw',
            'peak', 'mass', 'mean', 'std',
            'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
            'area', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw'])
    plt_array = []

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Update blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """
    for i in range(len(pims_frames)):
        current_frame = pims_frames[i]
        fnum = current_frame.frame_no

        current_blobs_df = blobs_df[blobs_df['frame'] == fnum]

        if isinstance(truth_df, pd.DataFrame):
            curr_truth_df = truth_df[truth_df['frame'] == fnum]
        else:
            curr_truth_df = None

        if isinstance(segm_df, pd.DataFrame):
            current_segm_df = segm_df[segm_df['frame'] == fnum]
        else:
            current_segm_df = None

        tmp_psf_df, tmp_plt_array = fit_psf(pims_frame=current_frame,
                       blobs_df=current_blobs_df,
                       diag_max_dist_err=diag_max_dist_err,
                       diag_max_sig_to_sigraw = diag_max_sig_to_sigraw,
                       diagnostic=diagnostic,
                       pltshow=pltshow,
                       truth_df=curr_truth_df,
                       segm_df=current_segm_df)
        df = pd.concat([df, tmp_psf_df], sort=False)
        plt_array.append(tmp_plt_array)

    psf_df = df
    plt_array = np.array(plt_array)

    return psf_df, plt_array
