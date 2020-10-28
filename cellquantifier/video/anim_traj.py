import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from ..plot.plotutil import *
from skimage.morphology import binary_dilation, binary_erosion, disk
from functools import reduce; import operator; import math

def anim_traj(df, tif,

            show_image=True,

            show_scalebar=True,
            pixel_size=None,
            scalebar_pos='upper right',
            scalebar_fontsize='large',
            scalebar_length=0.3,
            scalebar_height=0.02,
            scalebar_boxcolor=(1,1,1),
            scalebar_boxcolor_alpha=0,
            cb_fontsize='large',

            show_colorbar=True,
            cb_min=None,
            cb_max=None,
            cb_major_ticker=None,
            cb_minor_ticker=None,
            cb_pos='right',
            cb_tick_loc='right',

            plot_r=False,

            show_traj_num = False,
            fontname='Arial',

            show_tail=True,
            tail_length=50,

            show_boundary=False,
            boundary_masks=None,

            dpi=150,
            ):
    """
    Animate blob in the tif video.

    Pseudo code
    ----------
    1. If df is empty, return.
    2. Initialize fig, ax, and xlim, ylim based on df.
    3. Add scale bar.
    4. Animate blobs.

    Parameters
    ----------
    df : DataFrame
		DataFrame containing 'x', 'y', 'frame'.

    tif : Numpy array
        tif stack in format of 3d numpy array.

    pixel_size : None or float
        If None, no scalebar.
        If float, add scalebar with specified pixel size in um.

    scalebar_color : tuple
        RGB or RGBA tuple to define the scalebar color

    scalebar_pos : string
        position string. ('upper right'...)

    scalebar_length : float
        scalebar length relative to ax

    scalebar_height : float
        scalebar height relative to ax

    scalebar_boxcolor : tuple
        RGB or RGBA tuple to define the scalebar background box color

    scalebar_boxcolor_alpha : float
        Define the transparency of the background box color

    cb_fontsize : string or integer
        Define the colorbar label text size. eg. 'large', 'x-large', 10, 40...

    cb_min, cb_max: float
        [cb_min, cb_max] is the color bar range.

    cb_major_ticker, cb_minor_ticker: float
        Major and minor ticker setting for the color bar.

    cb_pos : string
        append_axes position string. ('right', 'top'...)

    cb_tick_loc : string
        colorbar tick position string. ('right', 'top'...)

    show_colorbar : bool
        If False, only return colormap and df, no colorbar added.

    show_image : bool
        if True, show animation of top of image.
        if False, only show blobs without image.

    tail_length : int
        how many frames will the traj last.

    dpi : int
        dpi setting for plt.subplots().


    Returns
    -------
    3d numpy array.

    Examples
	--------
    anim_tif = anim_traj(df, tif,
                pixel_size=0.163,
                show_image=True)
    imsave('anim-result.tif', anim_tif)
    """

    # """
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~
    # """
    if df.empty:
        print('df is empty. No traj to animate!')
        return

    anim_tif = []

    for i in range(len(tif)):
        print("Animate frame %d" % i)
        curr_df = df[ df['frame']==i ]

        # """
        # ~~~~~~~~Initialize fig, ax, and xlim, ylim based on df~~~~~~~~
        # """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.set_xticks([]); ax.set_yticks([])

        if not show_image:
            x_range = (df['y'].max() - df['y'].min())
            y_range = (df['x'].max() - df['x'].min())
            ax.set_xlim(df['y'].min()-0.1*x_range, df['y'].max()+0.1*x_range)
            ax.set_ylim(df['x'].max()+0.1*y_range, df['x'].min()-0.1*y_range)
        else:
            ax.imshow(tif[i], cmap='gray', aspect='equal')
            ax.set_xlim(0, tif[0].shape[1])
            ax.set_ylim(0, tif[0].shape[0])
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(False)

        # """
        # ~~~~~~~~Add scale bar~~~~~~~~
        # """
        if show_scalebar and pixel_size:
            add_scalebar(ax, units='um',
                        sb_color=(0.5,0.5,0.5),
                        fontname='Arial',
                        pixel_size=pixel_size,
                        sb_pos=scalebar_pos,
                        fontsize=scalebar_fontsize,
                        length_fraction=scalebar_length,
                        height_fraction=scalebar_height,
                        box_color=scalebar_boxcolor,
                        box_alpha=scalebar_boxcolor_alpha,
                        )

        # """
        # ~~~~~~~~customized the colorbar, then add it~~~~~~~~
        # """
        if 'D' in df:
            df, colormap = add_outside_colorbar(ax, df,
                        data_col='D',
                        cb_colormap='coolwarm',
                        label_font_size=cb_fontsize,
                        cb_min=cb_min,
                        cb_max=cb_max,
                        cb_major_ticker=cb_major_ticker,
                        cb_minor_ticker=cb_minor_ticker,
                        show_colorbar=show_colorbar,
                        label_str=r'D (nm$^2$/s)',
                        cb_pos=cb_pos,
                        cb_tick_loc=cb_tick_loc,
                        )
        else:
            df, colormap = add_outside_colorbar(ax, df,
                        data_col='particle',
                        cb_colormap='jet',
                        label_font_size=cb_fontsize,
                        cb_min=cb_min,
                        cb_max=cb_max,
                        cb_major_ticker=cb_major_ticker,
                        cb_minor_ticker=cb_minor_ticker,
                        show_colorbar=show_colorbar,
                        label_str='particle',
                        cb_pos=cb_pos,
                        cb_tick_loc=cb_tick_loc,
                        )

        # """
        # ~~~~~~~~Add traj num~~~~~~~~
        # """
        if show_traj_num:
            particles = curr_df.particle.unique()
            ax.text(0.95,
                    0.00,
                    """
                    Total trajectory number: %d
                    """ %(len(particles)),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize = 12,
                    color = (0.5, 0.5, 0.5, 0.5),
                    transform=ax.transAxes,
                    weight = 'bold',
                    fontname = fontname)

        # """
        # ~~~~~~~~Animate curr blob~~~~~~~~
        # Backup code for color coded blob animation:
        # for ind in curr_df.index:
        #     partcle_color = colormap(curr_df.loc[ind, 'D_norm'])
        #     y, x, r = curr_df.loc[ind, 'x'], curr_df.loc[ind, 'y'], curr_df.loc[ind, 'r']
        #     ax.scatter(x, y,
        #                 s=5,
        #                 marker='^',
        #                 c=[partcle_color])
        #     if False:
        #         c = plt.Circle((x,y), r, color=partcle_color,
        #                        linewidth=1, fill=False)
        #         ax.add_patch(c)
        # """
        anno_blob(ax, curr_df, marker='^', markersize=3, plot_r=plot_r,
                    color=(0,0,1))

        # """
        # ~~~~~~~~Animate particle tail~~~~~~~~
        # """
        if show_tail:
            particles = curr_df.particle.unique()
            for particle_num in particles:
                traj = df[df.particle == particle_num]
                traj = traj[ traj['frame'].isin(range(i-tail_length, i+1)) ]
                traj = traj.sort_values(by='frame')

                # """
                # ~~~~~~~~Animate cilia global trajectory if exists~~~~~~~~
                # """
                if 'x_global' in df.columns and 'y_global' in df.columns:
                    ax.plot(traj['y_global'], traj['x_global'], '-',
                            linewidth=0.5, color=(0,1,0))

                if 'D_norm' in traj:
                    ax.plot(traj['y'], traj['x'], linewidth=0.5,
                    			color=colormap(traj['D_norm'].mean()))
                else:
                    ax.plot(traj['y'], traj['x'], linewidth=0.5,
                    			color=colormap(traj['particle_norm'].mean()))

        # """
        # ~~~~~~~~Animate boundary~~~~~~~~
        # """
        if show_boundary:
            curr_mask = boundary_masks[i]
            selem = disk(1)
            bdr_pixels = curr_mask - binary_erosion(curr_mask, selem)
            bdr_coords = np.nonzero(bdr_pixels)
            ax.plot(bdr_coords[1], bdr_coords[0], 'o',
                            markersize=1,
                            linewidth=0.5,
                            color=(0,1,0,1))







        # """
        # ~~~~~~~~save curr figure~~~~~~~~
        # """
        curr_plt_array = plot_end(fig, pltshow=False)
        anim_tif.append(curr_plt_array)

    anim_tif = np.array(anim_tif)
    return anim_tif
