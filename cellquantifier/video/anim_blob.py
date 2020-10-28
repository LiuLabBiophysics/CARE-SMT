import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from ..plot.plotutil import *

def anim_blob(df, tif,
            pixel_size=None,
            scalebar_pos='upper right',
            scalebar_fontsize='large',
            scalebar_length=0.3,
            scalebar_height=0.02,
            scalebar_boxcolor=(1,1,1),
            scalebar_boxcolor_alpha=0,
            blob_markersize=3,
            plot_r=False,
            show_image=True,
            figsize=(9, 9),
            dpi=100,
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
		DataFrame containing 'x', 'y', 'r', 'frame'.

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

    show_image : bool
        if True, show animation of top of image.
        if False, only show blobs without image.

    dpi : int
        dpi setting for plt.subplots().

    Returns
    -------
    3d numpy array.

    Examples
	--------
    anim_tif = anim_blob(df, tif,
                pixel_size=0.163,
                show_image=True)
    imsave('anim-result.tif', anim_tif)
    """

    # """
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~
    # """
    if df.empty:
        print('df is empty. No blobs to animate!')
        return

    anim_tif = []
    for i in range(len(tif)):
        print("Animate frame %d" % i)
        curr_df = df[ df['frame']==i ]

        # """
        # ~~~~~~~~Initialize fig, ax, and xlim, ylim based on df~~~~~~~~
        # """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
        if pixel_size:
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
        # ~~~~~~~~Animate curr blob~~~~~~~~
        # """
        anno_blob(ax, curr_df, marker='^',
                    markersize=blob_markersize,
                    plot_r=plot_r,
                    color=(0,0,1))


        # """
        # ~~~~~~~~Animate curr blob~~~~~~~~
        # """
        ax.text(0.95,
                0.05,
                "Foci_num: %d" %(len(curr_df)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=12,
                color=(0.5, 0.5, 0.5, 0.5),
                transform=ax.transAxes,
                weight='bold',
                )


        # """
        # ~~~~~~~~save curr figure~~~~~~~~
        # """
        curr_plt_array = plot_end(fig, pltshow=False)
        anim_tif.append(curr_plt_array)

    anim_tif = np.array(anim_tif)

    return anim_tif
