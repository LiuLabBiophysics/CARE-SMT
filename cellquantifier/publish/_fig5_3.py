import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from ..plot.plotutil import *


def plot_fig5_3():

    # """
	# ~~~~~~~~~~~Prepare the data~~~~~~~~~~~~~~
	# """
    df = pd.read_csv('/home/linhua/Desktop/temp/200211_CtrBLM-physDataMerged.csv')
    df.loc[ df['sort_flag_53bp1']==0, 'sort_flag_53bp1'] = 'Far from 53bp1'
    df.loc[ df['sort_flag_53bp1']==1, 'sort_flag_53bp1' ] = 'Near 53bp1'
    df['A_to_area'] = df['A'] / df['area']

    df_non53bp1 = df[ df['sort_flag_53bp1']=='Far from 53bp1' ]
    df_53bp1 = df[ df['sort_flag_53bp1']=='Near 53bp1' ]

    # """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2_1 = ax.inset_axes([0, 0, 0.2, 0.2])
    ax2_2 = ax.inset_axes([0.3, 0, 0.2, 0.2])
    ax2_3 = ax.inset_axes([0.6, 0, 0.2, 0.2])
    ax1_1 = ax.inset_axes([0, 0.25, 0.2, 0.2])
    ax1_2 = ax.inset_axes([0.3, 0.25, 0.2, 0.2])
    ax1_3 = ax.inset_axes([0.6, 0.25, 0.2, 0.2])



    # """
	# ~~~~~~~~~~~Add plots to the grid~~~~~~~~~~~~~~
	# """

    # plot 2_1
    add_violin_2(ax2_1,
                df=df_53bp1,
                data_col='A',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax2_1,
                blobs_df=df_53bp1,
                cat_col='exp_label',
                hist_col='A',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax2_1,
                xlabel='',
                ylabel='Foci peak intensity (ADU)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-10, 300, 50],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax2_1.get_xaxis().set_ticks([])

    # plot 2_2
    add_violin_2(ax2_2,
                df=df_53bp1,
                data_col='area',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax2_2,
                blobs_df=df_53bp1,
                cat_col='exp_label',
                hist_col='area',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax2_2,
                xlabel='',
                ylabel=r'Foci area (pixel$^2$)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-1, 30, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax2_2.get_xaxis().set_ticks([])


    # plot 2_3
    add_violin_2(ax2_3,
                df=df_53bp1,
                data_col='A_to_area',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax2_3,
                blobs_df=df_53bp1,
                cat_col='exp_label',
                hist_col='A_to_area',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax2_3,
                xlabel='',
                ylabel=r'Compactness (ADU / pixel$^2$)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-1, 30, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax2_3.get_xaxis().set_ticks([])


    # plot 1_1
    add_violin_2(ax1_1,
                df=df_non53bp1,
                data_col='A',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax1_1,
                blobs_df=df_non53bp1,
                cat_col='exp_label',
                hist_col='A',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax1_1,
                xlabel='',
                ylabel='Foci peak intensity (ADU)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-10, 300, 50],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax1_1.get_xaxis().set_ticks([])

    # plot 1_2
    add_violin_2(ax1_2,
                df=df_non53bp1,
                data_col='area',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax1_2,
                blobs_df=df_non53bp1,
                cat_col='exp_label',
                hist_col='area',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax1_2,
                xlabel='',
                ylabel=r'Foci area (pixel$^2$)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-1, 30, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax1_2.get_xaxis().set_ticks([])


    # plot 1_3
    add_violin_2(ax1_3,
                df=df_non53bp1,
                data_col='A_to_area',
                cat_col='exp_label',
                hue_order=['Ctr', 'BLM']
                )

    add_t_test(ax1_3,
                blobs_df=df_non53bp1,
                cat_col='exp_label',
                hist_col='A_to_area',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=10
                )

    format_ax(ax1_3,
                xlabel='',
                ylabel=r'Compactness (ADU / pixel$^2$)',
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[-1, 30, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=10,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Arial',
                legend_fontweight='normal',
                legend_fontsize=10)
    ax1_3.get_xaxis().set_ticks([])

    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Fig5_3.pdf')
    import webbrowser
    webbrowser.open_new(r'/home/linhua/Desktop/Fig5_3.pdf')
    plt.clf(); plt.close()
