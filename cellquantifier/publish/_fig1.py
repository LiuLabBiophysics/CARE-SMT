import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from ..plot.plotutil import *


def plot_fig_1(df):

    # """
	# ~~~~~~~~~~~Prepare the data~~~~~~~~~~~~~~
	# """
    df = pd.read_csv('/home/linhua/Desktop/dutp_paper/fig1/BLM2.csv')
    img = imread('/home/linhua/Desktop/dutp_paper/fig1/BLM2.tif')[0]
    df = df[ df['traj_length']>50 ]
    img_53bp1 = imread('/home/linhua/Desktop/dutp_paper/fig1/53bp1channel.jpg')
    img_dutp = imread('/home/linhua/Desktop/dutp_paper/fig1/dutpchannel.jpg')
    img_comb = imread('/home/linhua/Desktop/dutp_paper/fig1/composite.jpg')


    # """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_xticks([]); ax.set_yticks([])
    ax3_1 = ax.inset_axes([0, 0, 0.2, 0.2])
    ax3_2 = ax.inset_axes([0.3, 0, 0.2, 0.2])
    ax3_3 = ax.inset_axes([0.6, 0, 0.2, 0.2])
    ax2_4 = ax.inset_axes([0.16, 0.206, 0.498, 0.498])
    ax2_4a = ax2_4.inset_axes([0.775, 0.0, 0.3, 0.3]); ax2_4a.set_anchor('SW')
    ax2_4b = ax2_4.inset_axes([0.0, 0.70, 0.3, 0.3]); ax2_4b.set_anchor('SW')
    ax2_3 = ax.inset_axes([0, 0.22, 0.15, 0.15])
    ax2_2 = ax.inset_axes([0, 0.38, 0.15, 0.15])
    ax2_1 = ax.inset_axes([0, 0.54, 0.15, 0.15])


    # """
	# ~~~~~~~~~~~Add plots to the grid~~~~~~~~~~~~~~
	# """
    ax2_1.imshow(img_53bp1, aspect='equal')
    ax2_1.set_xticks([]); ax2_1.set_yticks([])
    ax2_1.text(0.01, 0.98, '53BP1',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize = 20,
            color = (1, 1, 1),
            transform=ax2_1.transAxes,
            weight = 'bold',
            fontname = 'Arial')


    ax2_2.imshow(img_dutp, aspect='equal')
    ax2_2.set_xticks([]); ax2_2.set_yticks([])
    ax2_2.text(0.01, 0.98, 'DUTP',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize = 20,
            color = (1, 1, 1),
            transform=ax2_2.transAxes,
            weight = 'bold',
            fontname = 'Arial')

    ax2_3.imshow(img_comb, aspect='equal')
    ax2_3.set_xticks([]); ax2_3.set_yticks([])
    ax2_3.text(0.01, 0.98, 'Merged',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize = 20,
            color = (1, 1, 1),
            transform=ax2_3.transAxes,
            weight = 'bold',
            fontname = 'Arial')

    anno_traj(ax2_4, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                scalebar_pos='upper right',
                scalebar_fontsize=20,
                scalebar_length=0.4,
                scalebar_height=0.03,
                cb_fontsize=20,
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=None,
                show_colorbar=True)


    anno_traj(ax2_4a, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                scalebar_pos='lower left',
                scalebar_fontsize=15,
                scalebar_length=0.35,
                scalebar_height=0.05,
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=85,
                show_colorbar=False)


    anno_traj(ax2_4b, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                scalebar_pos='lower left',
                scalebar_fontsize=15,
                scalebar_length=0.25,
                scalebar_height=0.05,
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=55,
                show_colorbar=False)

    # plot 3_1
    add_mean_msd(ax3_1, df,
                cat_col=None,
                pixel_size=0.108,
                frame_rate=3.33,
                divide_num=5,
                RGBA_alpha=0.8,
                fitting_linewidth=1,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)
    format_ax(ax3_1,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-0.5, 12.5, 4],
                yscale=[3000, 14000, 4000],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=19,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=19)

    # plot 3_2
    add_D_hist(ax3_2, df,
                cat_col=None,
                RGBA_alpha=0.5,
                set_format=False)
    format_ax(ax3_2,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Frequency (a.u)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 4000, 2000],
                yscale=[0, 0.0008, 0.0004],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=19,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=19)
    ax3_2.set_yticklabels([-0.5, 0, 0.5, 1])

    # plot 3_3
    add_alpha_hist(ax3_3, df,
                cat_col=None,
                RGBA_alpha=0.5,
                set_format=False)
    format_ax(ax3_3,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Frequency (a.u)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.8, 0.4],
                yscale=[0, 4, 2],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=19,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=19)
    ax3_3.set_yticklabels([-0.5, 0, 0.5, 1])


    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    import webbrowser
    webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
