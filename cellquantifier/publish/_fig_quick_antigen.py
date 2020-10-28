import pandas as pd; import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ..phys.physutil import *
from ..plot.plotutil import *


def fig_quick_antigen(df=pd.DataFrame([])):
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty:
        df = df[ df['traj_length'] > 25 ]

        df['traj_length'] = df['traj_length'] * 0.5
        df['dist_to_boundary'] = df['dist_to_boundary'] * -1
        df['v'] = df['v'] * 2
        df['v_max'] = df['v_max'] * 2


        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')





    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(8.5, 11))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.67, 1, 0.33])
    # fig2 = left_page.inset_axes([0, 0.56, 1, 0.33])
    # fig3 = left_page.inset_axes([0, 0.23, 1, 0.33])
    #
    fig1_1 = fig1.inset_axes([0.13, 0.78, 0.3, 0.2])
    # fig1_2 = fig1.inset_axes([0.6, 0.78, 0.3, 0.2])
    # fig1_3 = fig1.inset_axes([0.13, 0.48, 0.3, 0.2])
    # fig1_4 = fig1.inset_axes([0.6, 0.48, 0.3, 0.2])
    # fig1_5 = fig1.inset_axes([0.13, 0.18, 0.3, 0.2])







    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1]:
        axis.set_xticks([]); axis.set_yticks([])

    # """
	# ~~~~Plot fig1~~~~
	# """
    # Plot fig1_1: dist_to_boundary
    add_hist(fig1_1,
                df=df_particle,
                data_col='dist_to_boundary',
                hist_kws={'linewidth':0.5},
                kde=False
                )

    format_ax(fig1_1,
                xlabel='dist_to_boundary (um)',
                ylabel='counts',
                spine_linewidth=0.5,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 10, 2, 1],
                yscale=[0, 50, 10, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                tk_width=0.1,
                show_legend=False,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_2.set_yticklabels([-0.5, 0, 0.5, 1])

    # # Plot fig1_2:
    # add_hist(fig1_2,
    #             df=df_particle,
    #             data_col='dist_to_53bp1',
    #             hist_kws={'linewidth':0.5},
    #             kde=False
    #             )
    #
    # format_ax(fig1_2,
    #             xlabel='dist_to_nucleus (um)',
    #             ylabel='counts',
    #             spine_linewidth=1,
    #             xlabel_color=(0,0,0,1),
    #             ylabel_color=(0,0,0,1),
    #             xscale=[0, 40, 10],
    #             yscale=[0, 75, 25],
    #             label_fontname='Arial',
    #             label_fontweight='normal',
    #             label_fontsize=9,
    #             tklabel_fontname='Arial',
    #             tklabel_fontweight='normal',
    #             tklabel_fontsize=8,
    #             show_legend=False,
    #             legend_loc='upper right',
    #             legend_frameon=False,
    #             legend_fontname='Liberation Sans',
    #             legend_fontweight=6,
    #             legend_fontsize=6)
    # # fig1_2.set_yticklabels([-0.5, 0, 0.5, 1])
    #
    # # Plot fig1_3:
    # add_hist(fig1_3,
    #             df=df_particle,
    #             data_col='travel_dist',
    #             hist_kws={'linewidth':0.5},
    #             kde=False
    #             )
    #
    # format_ax(fig1_3,
    #             xlabel='travel_dist (um)',
    #             ylabel='counts',
    #             spine_linewidth=1,
    #             xlabel_color=(0,0,0,1),
    #             ylabel_color=(0,0,0,1),
    #             xscale=[0, 5, 1],
    #             yscale=[0, 55, 10],
    #             label_fontname='Arial',
    #             label_fontweight='normal',
    #             label_fontsize=9,
    #             tklabel_fontname='Arial',
    #             tklabel_fontweight='normal',
    #             tklabel_fontsize=8,
    #             show_legend=False,
    #             legend_loc='upper right',
    #             legend_frameon=False,
    #             legend_fontname='Liberation Sans',
    #             legend_fontweight=6,
    #             legend_fontsize=6)
    # # fig1_3.set_yticklabels([-0.5, 0, 0.5, 1])
    #
    # # Plot fig1_5:
    # add_hist(fig1_5,
    #             df=df_particle,
    #             data_col='v_max',
    #             hist_kws={'linewidth':0.5},
    #             kde=False
    #             )
    #
    # format_ax(fig1_5,
    #             xlabel='max speed (um/s)',
    #             ylabel='counts',
    #             spine_linewidth=1,
    #             xlabel_color=(0,0,0,1),
    #             ylabel_color=(0,0,0,1),
    #             xscale=[0, 1, 0.5],
    #             yscale=[0, 50, 10],
    #             label_fontname='Arial',
    #             label_fontweight='normal',
    #             label_fontsize=9,
    #             tklabel_fontname='Arial',
    #             tklabel_fontweight='normal',
    #             tklabel_fontsize=8,
    #             show_legend=False,
    #             legend_loc='upper right',
    #             legend_frameon=False,
    #             legend_fontname='Liberation Sans',
    #             legend_fontweight=6,
    #             legend_fontsize=6)
    #
    # # Plot fig1_4:
    # add_hist(fig1_4,
    #             df=df_particle,
    #             data_col='traj_length',
    #             hist_kws={'linewidth':0.5},
    #             kde=False
    #             )
    #
    # format_ax(fig1_4,
    #             xlabel='life time (s)',
    #             ylabel='counts',
    #             spine_linewidth=1,
    #             xlabel_color=(0,0,0,1),
    #             ylabel_color=(0,0,0,1),
    #             xscale=[0, 300, 100],
    #             # yscale=[0, 100, 25],
    #             label_fontname='Arial',
    #             label_fontweight='normal',
    #             label_fontsize=9,
    #             tklabel_fontname='Arial',
    #             tklabel_fontweight='normal',
    #             tklabel_fontsize=8,
    #             show_legend=False,
    #             legend_loc='upper right',
    #             legend_frameon=False,
    #             legend_fontname='Liberation Sans',
    #             legend_fontweight=6,
    #             legend_fontsize=6)









    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    # plt.clf(); plt.close()
