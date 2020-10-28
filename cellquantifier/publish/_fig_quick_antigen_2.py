import pandas as pd; import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ..phys.physutil import *
from ..plot.plotutil import *


def fig_quick_antigen_2(df=pd.DataFrame([])):
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty:
        # """
        # ~~~~~~~~traj_length filter~~~~~~~~
        # """
        if 'traj_length' in df:
            df = df[ df['traj_length']>=50 ]

        # """
        # ~~~~~~~~travel_dist filter~~~~~~~~
        # """
        if 'travel_dist' in df:
            travel_dist_min = 0
            travel_dist_max = 7
            df = df[ (df['travel_dist']>=travel_dist_min) & \
            					(df['travel_dist']<=travel_dist_max) ]

        # """
        # ~~~~~~~~add particle type filter~~~~~~~~
        # """
        if 'particle type' in df:
        	df = df[ df['particle type']!='--none--']




        df['date'] = df['raw_data'].astype(str).str[0:6]
        df_mal = df[ df['date'].isin(['200205']) ]
        df_mal_A = df_mal[ df_mal['particle type']=='A' ]
        df_mal_B = df_mal[ df_mal['particle type']=='B' ]
        df_cas9 = df[ df['date'].isin(['200220']) ]
        df_cas9_A = df_cas9[ df_cas9['particle type']=='A' ]
        df_cas9_B = df_cas9[ df_cas9['particle type']=='B' ]

        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')

        df_mal_particle = df_mal.drop_duplicates('particle')
        df_mal_A_particle = df_mal_A.drop_duplicates('particle')
        df_mal_B_particle = df_mal_B.drop_duplicates('particle')

        df_cas9_particle = df_cas9.drop_duplicates('particle')
        df_cas9_A_particle = df_cas9_A.drop_duplicates('particle')
        df_cas9_B_particle = df_cas9_B.drop_duplicates('particle')




    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(17, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.67, 1, 0.33])
    fig2 = left_page.inset_axes([0, 0.56, 1, 0.11])
    fig3 = left_page.inset_axes([0, 0.45, 1, 0.11])

    fig1_1 = fig1.inset_axes([0.13, 0.78, 0.2, 0.2])
    fig1_2 = fig1.inset_axes([0.45, 0.78, 0.2, 0.2])
    fig1_3 = fig1.inset_axes([0.77, 0.78, 0.2, 0.2])
    fig1_4 = fig1.inset_axes([0.13, 0.48, 0.2, 0.2])
    fig1_5 = fig1.inset_axes([0.45, 0.48, 0.2, 0.2])
    fig1_6 = fig1.inset_axes([0.77, 0.48, 0.2, 0.2])
    fig1_7 = fig1.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig1_8 = fig1.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig1_9 = fig1.inset_axes([0.77, 0.18, 0.2, 0.2])

    fig2_1 = fig2.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig2_2 = fig2.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig3_1 = fig3.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig3_2 = fig3.inset_axes([0.6, 0.35, 0.3, 0.6])



    fig4 = right_page.inset_axes([0, 0.67, 1, 0.33])
    fig5 = right_page.inset_axes([0, 0.56, 1, 0.11])
    fig6 = right_page.inset_axes([0, 0.45, 1, 0.11])

    fig4_1 = fig4.inset_axes([0.13, 0.78, 0.2, 0.2])
    fig4_2 = fig4.inset_axes([0.45, 0.78, 0.2, 0.2])
    fig4_3 = fig4.inset_axes([0.77, 0.78, 0.2, 0.2])
    fig4_4 = fig4.inset_axes([0.13, 0.48, 0.2, 0.2])
    fig4_5 = fig4.inset_axes([0.45, 0.48, 0.2, 0.2])
    fig4_6 = fig4.inset_axes([0.77, 0.48, 0.2, 0.2])
    fig4_7 = fig4.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig4_8 = fig4.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig4_9 = fig4.inset_axes([0.77, 0.18, 0.2, 0.2])

    fig5_1 = fig5.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig5_2 = fig5.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig6_1 = fig6.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig6_2 = fig6.inset_axes([0.6, 0.35, 0.3, 0.6])




    color_list = [plt.cm.get_cmap('coolwarm')(0),
                    plt.cm.get_cmap('coolwarm')(0.5),
                    plt.cm.get_cmap('coolwarm')(0.99),
                    ]
    # color_list = [(0,0,1),(0,0,0),(1,0,0)]


    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1, fig2, fig3, fig4, fig5, fig6]:
        axis.set_xticks([]); axis.set_yticks([])

    # """
	# ~~~~Plot fig1~~~~
	# """
    # Plot fig1_1: mean msd curve
    add_mean_msd(fig1_1, df_mal,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig1_1,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig1_2: D
    add_D_hist(fig1_2,
                df=df_mal_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_2,
                blobs_df=df_mal_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_2,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_2.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig1_3: alpha
    add_alpha_hist(fig1_3,
                df=df_mal_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_3,
                blobs_df=df_mal_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_3,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_3.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig1_4: mean msd curve
    add_mean_msd(fig1_4, df_mal_A,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig1_4,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig1_5: D
    add_D_hist(fig1_5,
                df=df_mal_A_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_5,
                blobs_df=df_mal_A_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_5,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_5.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig1_6: alpha
    add_alpha_hist(fig1_6,
                df=df_mal_A_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_6,
                blobs_df=df_mal_A_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_6,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_6.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig1_7: mean msd curve
    add_mean_msd(fig1_7, df_mal_B,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig1_7,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig1_8: D
    add_D_hist(fig1_8,
                df=df_mal_B_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_8,
                blobs_df=df_mal_B_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_8,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_8.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig1_9: alpha
    add_alpha_hist(fig1_9,
                df=df_mal_B_particle,
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig1_9,
                blobs_df=df_mal_B_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig1_9,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig1_9.set_yticklabels([-0.5, 0, 0.5, 1])


    # """
	# ~~~~Plot fig2~~~~
	# """
    # Plot fig2_1:
    add_hist(fig2_1,
                df=df_mal_A_particle,
                data_col='lifetime',
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    format_ax(fig2_1,
                xlabel='lifetime (frame)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)

    # Plot fig2_2:
    add_hist(fig2_2,
                df=df_mal_B_particle,
                data_col='lifetime',
                cat_col='exp_label',
                cat_order=['MalKN', 'WT', 'MalOE'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    format_ax(fig2_2,
                xlabel='lifetime (frame)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)


    # """
	# ~~~~Plot fig4~~~~
	# """
    # Plot fig4_1: mean msd curve
    add_mean_msd(fig4_1, df_cas9,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig4_1,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig4_2: D
    add_D_hist(fig4_2,
                df=df_cas9_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_2,
                blobs_df=df_cas9_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_2,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_2.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig4_3: alpha
    add_alpha_hist(fig4_3,
                df=df_cas9_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_3,
                blobs_df=df_cas9_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_3,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_3.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig4_4: mean msd curve
    add_mean_msd(fig4_4, df_cas9_A,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig4_4,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig4_5: D
    add_D_hist(fig4_5,
                df=df_cas9_A_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_5,
                blobs_df=df_cas9_A_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_5,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_5.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig4_6: alpha
    add_alpha_hist(fig4_6,
                df=df_cas9_A_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_6,
                blobs_df=df_cas9_A_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_6,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_6.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig4_7: mean msd curve
    add_mean_msd(fig4_7, df_cas9_B,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                pixel_size=0.163,
                frame_rate=2,
                divide_num=5,
                RGBA_alpha=0.5,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig4_7,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[-2, 34, 10],
                # yscale=[-10000, 300000, 100000],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig4_8: D
    add_D_hist(fig4_8,
                df=df_cas9_B_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_8,
                blobs_df=df_cas9_B_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_8,
                xlabel=r'D (nm$^2$/s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_8.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig4_9: alpha
    add_alpha_hist(fig4_9,
                df=df_cas9_B_particle,
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=True,
                set_format=False,
                )

    add_t_test(fig4_9,
                blobs_df=df_cas9_B_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig4_9,
                xlabel=r'$\mathit{\alpha}$',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 2, 1],
                # yscale=[0, 15, 5],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
    # fig4_9.set_yticklabels([-0.5, 0, 0.5, 1])

    # """
	# ~~~~Plot fig5~~~~
	# """
    # Plot fig5_1:
    add_hist(fig5_1,
                df=df_cas9_A_particle,
                data_col='lifetime',
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    format_ax(fig5_1,
                xlabel='lifetime (frame)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)

    # Plot fig5_2:
    add_hist(fig5_2,
                df=df_cas9_B_particle,
                data_col='lifetime',
                cat_col='exp_label',
                cat_order=['Cas9P3', 'Cas9L1', 'Cas9C1'],
                color_list=color_list,
                RGBA_alpha=0.5,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    format_ax(fig5_2,
                xlabel='lifetime (frame)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                # xscale=[0, 1000, 500],
                # yscale=[0, 9, 4],
                label_fontname='Arial',
                label_fontweight='normal',
                label_fontsize=9,
                tklabel_fontname='Arial',
                tklabel_fontweight='normal',
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='upper right',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6)
















    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
