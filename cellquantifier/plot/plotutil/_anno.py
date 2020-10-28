import math
import numpy as np
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import trackpy as tp
from ._add_colorbar import add_outside_colorbar
from ._add_scalebar import add_scalebar
from skimage.morphology import binary_dilation, binary_erosion, disk


def set_ylim_reverse(ax):
    """
    This function is needed for annotation. Since ax.imshow(img) display
    the img in a different manner comparing with traditional axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    """
    bottom, top = ax.get_ylim()
    if top > bottom:
        ax.set_ylim(top, bottom)

def anno_ellipse(ax, regionprops, linewidth=1, color=(1,0,0,0.8)):
    """
    Annotate ellipse in matplotlib axis.
    The ellipse parameters are obtained from regionprops object of skimage.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    regionprops : list of object
        Measure properties of labeled image regions.
        regionprops is the return value of skimage.measure.regionprops().
    linewidth: float, optional
        Linewidth of the ellipse.
    color: tuple, optional
        color of the ellipse.

    Returns
    -------
    Annotate edge, long axis, short axis of ellipses.
    """

    set_ylim_reverse(ax)

    for region in regionprops:
        row, col = region.centroid
        y0, x0 = row, col
        orientation = region.orientation
        ax.plot(x0, y0, '.', markersize=15, color=color)
        x1 = x0 + math.cos(orientation) * 0.5 * region.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * region.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * region.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * region.major_axis_length
        ax.plot((x0, x1), (y0, y1), '-', linewidth=linewidth, color=color)
        ax.plot((x0, x2), (y0, y2), '-', linewidth=linewidth, color=color)
        curr_e = patches.Ellipse((x0, y0), width=region.minor_axis_length,
                        height=region.major_axis_length,
                        angle=-orientation/math.pi*180, facecolor='None',
                        linewidth=linewidth, edgecolor=color)
        ax.add_patch(curr_e)

def anno_blob(ax, blob_df,
            marker='s',
            markersize=10,
            plot_r=True,
            color=(0,1,0,0.8)):
    """
    Annotate blob in matplotlib axis.
    The blob parameters are obtained from blob_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    blob_df : DataFrame
        bolb_df has columns of 'x', 'y', 'r'.
    makers: string, optional
        The marker for center of the blob.
    plot_r: bool, optional
        If True, plot the circle.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate center and the periphery of the blob.
    """

    set_ylim_reverse(ax)

    f = blob_df
    for i in f.index:
        y, x, r = f.at[i, 'x'], f.at[i, 'y'], f.at[i, 'r']
        ax.scatter(x, y,
                    s=markersize,
                    marker=marker,
                    c=[color])
        if plot_r:
            c = plt.Circle((x,y), r, color=color,
                           linewidth=1, fill=False)
            ax.add_patch(c)

def anno_scatter(ax, scatter_df, marker = 'o', color=(0,1,0,0.8)):
    """
    Annotate scatter in matplotlib axis.
    The scatter parameters are obtained from scatter_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    scatter_df : DataFrame
        scatter_df has columns of 'x', 'y'.
    makers: string, optional
        The marker for the position of the scatter.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate scatter in the ax.
    """

    set_ylim_reverse(ax)

    f = scatter_df
    for i in f.index:
        y, x = f.at[i, 'x'], f.at[i, 'y']
        ax.scatter(x, y,
                    s=10,
                    marker=marker,
                    c=[color])


def anno_traj(ax, df,

            show_image=True,
            image=np.array([]),

            show_scalebar=True,
            pixel_size=None,
            scalebar_pos='upper right',
            scalebar_fontsize='large',
            scalebar_length=0.3,
            scalebar_height=0.02,
            scalebar_boxcolor=(1,1,1),
            scalebar_boxcolor_alpha=0,

            show_colorbar=True,
            cb_fontsize='large',
            cb_min=None,
            cb_max=None,
            cb_major_ticker=None,
            cb_minor_ticker=None,
            cb_pos='right',
            cb_tick_loc='right',

            show_traj_num=True,
            fontname='Arial',

            show_traj_end=False,

            show_particle_label=False,
            choose_particle=None,

            show_boundary=False,
            boundary_mask=None,
            boundary_list=None,
            ):
    """
    Annotate trajectories in matplotlib axis.
    The trajectories parameters are obtained from blob_df.
    The colorbar locates "outside" of the traj figure.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate trajectories.

    df : DataFrame
		DataFrame containing 'D', 'frame', 'x', and 'y' columns

	image: 2D ndarray
		The image the trajectories will be plotted on

    pixel_size: float
		The pixel_size of the images in microns/pixel

    scalebar_pos: string
        string for scalebar position. like 'upper right', 'lower right' ...

    show_traj_num: bool
        If true, show a text with trajectory number

    fontname: string
        Font used in the figure. Default is 'Arial'

    cb_min, cb_max: float
        [cb_min, cb_max] is the color bar range.

    cb_major_ticker, cb_minor_ticker: float
        Major and minor setting for the color bar

    show_particle_label : bool
        If true, add particle label in the figure.

    choose_particle : None or integer
        If particle number specifier, only plot that partcle.

    Returns
    -------
    Annotate trajectories in the ax.
    """
    # """
    # ~~~~~~~~~~~If choose_particle is True, prepare df and image~~~~~~~~~~~~~~
    # """
    original_df = df.copy()
    if choose_particle != None:
        df = df[ df['particle']==choose_particle ]

        r_min = int(round(df['x'].min()))
        c_min = int(round(df['y'].min()))
        delta_x = int(round(df['x'].max()) - round(df['x'].min()))
        delta_y = int(round(df['y'].max()) - round(df['y'].min()))
        delta = int(max(delta_x, delta_y))
        if delta < 1:
            delta = 1
        r_max = r_min + delta_x
        c_max = c_min + delta_y

        df['x'] = df['x'] - r_min
        df['y'] = df['y'] - c_min
        # print('#############################')
        # print(df['x'].min(), df['y'].min())
        # print(df['x'].max(), df['y'].max())
        # print('#############################')

        image = image[r_min:r_max+1, c_min:c_max+1]
    # """
    # ~~~~~~~~~~~Check if df is empty. Plot the image if True~~~~~~~~~~~~~~
    # """
    if df.empty:
    	return

    if show_image and image.size != 0:
        ax.imshow(image, cmap='gray', aspect='equal')
        plt.box(False)


    # """
    # ~~~~~~~~~~~Add pixel size scale bar~~~~~~~~~~~~~~
    # """
    if show_scalebar and pixel_size:
        add_scalebar(ax, pixel_size=pixel_size, units='um',
                    sb_color=(0.5,0.5,0.5),
                    sb_pos=scalebar_pos,
                    length_fraction=scalebar_length,
                    height_fraction=scalebar_height,
                    box_color=scalebar_boxcolor,
                    box_alpha=scalebar_boxcolor_alpha,
                    fontname='Arial',
                    fontsize=scalebar_fontsize)


    # """
    # ~~~~~~~~~~~customized the colorbar, then add it~~~~~~~~~~~~~~
    # """
    if 'D' in original_df:
        modified_df, colormap = add_outside_colorbar(ax, original_df,
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
                    cb_tick_loc=cb_tick_loc)
    else:
        modified_df, colormap = add_outside_colorbar(ax, original_df,
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
                    cb_tick_loc=cb_tick_loc)


    # """
    # ~~~~~~~~~~~Plot the color coded trajectories using colorbar norm~~~~~~~~~~~~~~
    # """
    if choose_particle:
        df['D_norm'] = modified_df[ modified_df['particle']== choose_particle ]['D_norm']
    else:
        df = modified_df

    ax.set_aspect(1.0)
    particles = df.particle.unique()
    for particle_num in particles:
        traj = df[df.particle == particle_num]
        traj = traj.sort_values(by='frame')
        traj_start = traj.head(1)
        traj_end = traj.tail(1)

        # """
        # ~~~Plot global movement if 'x_global', 'y_global' in df.columns~~~
        # """
        if 'x_global' in df.columns and 'y_global' in df.columns:
            ax.plot(traj['y_global'], traj['x_global'], '-',
                    linewidth=1, color=(0,1,0))
            ax.plot(traj['y'], traj['x'], 'o', linewidth=1,
            			color=colormap(traj['D_norm'].mean()))
            for ind in traj.index:
                temp = np.zeros((2, 2))
                temp[0,:] = df.loc[ind, ['x', 'y']].to_numpy()
                temp[1,:] = df.loc[ind, ['x_global', 'y_global']].to_numpy()
                if 'half_sign' in traj.columns:
                    if traj.loc[ind, 'half_sign'] < 0:
                        ax.plot(temp[:,1], temp[:,0], color=(0,0,1))
                    else:
                        ax.plot(temp[:,1], temp[:,0], color=(1,0,0))

                else:
                    ax.plot(temp[:,1], temp[:,0],
                            color=colormap(traj['D_norm'].mean()))
        else:
            if 'D_norm' in traj:
                ax.plot(traj['y'], traj['x'], linewidth=1,
                			color=colormap(traj['D_norm'].mean()))
                if show_traj_end:
                    ax.plot(traj_end['y'], traj_end['x'], '^',
                            markersize=3, fillstyle='none',
                			color=colormap(traj['D_norm'].mean()))
            else:
                ax.plot(traj['y'], traj['x'], linewidth=1,
                			color=colormap(traj['particle_norm'].mean()))
                if show_traj_end:
                    ax.plot(traj_end['y'], traj_end['x'], '^',
                            markersize=3, fillstyle='none',
                			color=colormap(traj['particle_norm'].mean()))

        if show_particle_label:
            ax.text(traj['y'].mean(), traj['x'].mean(),
                    particle_num, color=(0, 1, 0))

    if show_traj_num:
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
    # ~~~~~~~~Draw boundary~~~~~~~~
    # """
    if show_boundary:
        if boundary_list == None:
            selem = disk(1)
            bdr_pixels = boundary_mask ^ binary_erosion(boundary_mask, selem)
            bdr_coords = np.nonzero(bdr_pixels)
            ax.plot(bdr_coords[1], bdr_coords[0], 'o',
                            markersize=1,
                            linewidth=0.5,
                            color=(0,1,0,0.5))
        else:
            selem_in = disk(np.abs(boundary_list[0]))
            if boundary_list[0] < 0:
                bdr_mask_1 = binary_erosion(boundary_mask, selem_in)
            else:
                bdr_mask_1 = binary_dilation(boundary_mask, selem_in)
            bdr_pixels_1 = bdr_mask_1 ^ binary_erosion(bdr_mask_1, disk(1))
            bdr_coords_1 = np.nonzero(bdr_pixels_1)
            ax.plot(bdr_coords_1[1], bdr_coords_1[0], 'o',
                            markersize=1,
                            linewidth=0.5,
                            color=(0,1,0,0.5))

            selem_out = disk(np.abs(boundary_list[1]))
            if boundary_list[1] < 0:
                bdr_mask_2 = binary_erosion(boundary_mask, selem_out)
            else:
                bdr_mask_2 = binary_dilation(boundary_mask, selem_out)
            bdr_pixels_2 = bdr_mask_2 ^ binary_erosion(bdr_mask_2, disk(1))
            bdr_coords_2 = np.nonzero(bdr_pixels_2)
            ax.plot(bdr_coords_2[1], bdr_coords_2[0], 'o',
                            markersize=1,
                            linewidth=0.5,
                            color=(0,1,0,0.5))



    # """
    # ~~~~~~~~~~~Set ax format~~~~~~~~~~~~~~
    # """
    set_ylim_reverse(ax)
    ax.set_xticks([])
    ax.set_yticks([])

def anno_raw(im, df, color=(255,0,0)):

    """
    Annotate scatter on the original image or movie
    The scatter coordinates are obtained from df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    df : DataFrame
        df has columns of 'x', 'y'.
    color: tuple, optional
        Color of the marker and circle.

    """

    from skimage.io import imsave
    im = np.array(im)
    im_rgb = []

    for frame in df['frame'].unique():
        this_df = df.loc[df['frame'] == frame]
        this_frame = im[frame]
        this_frame_rgb = np.dstack((this_frame, this_frame, this_frame))
        for i in this_df.index:
            y, x = int(this_df.at[i, 'x']), int(this_df.at[i, 'y'])
            this_frame_rgb[y, x, :] = np.array(color)
        im_rgb.append(this_frame_rgb)

    im_rgb = np.array(im_rgb)

    return im_rgb
