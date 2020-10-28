import pandas as pd; import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ..phys.physutil import *
from ..plot.plotutil import *


def fig_quick_cilia(df=pd.DataFrame([])):
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty:
        df.loc[ df['v_norm']>0, 'direction'] = 'toward to tip'
        df.loc[ df['v_norm']<=0, 'direction' ] = 'toward to base'
        df.loc[ df['h_norm']>=0.8, 'location'] = 'tip'
        df.loc[ df['h_norm']<=0.2, 'location'] = 'base'
        df.loc[ (df['h_norm']>0.2)&(df['h_norm']<0.8), 'location' ] = 'middle'

        # add 'dist_to_base', bin_size=0.1
        df['dist_to_base'] = round(df['h_norm'], 1)

        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')
        df_dropna = df.dropna()
        df_dropna_a1 = df_dropna[ (df_dropna['exp_label']=='cohort a') & \
                            (df_dropna['location']!='base')]
        df_dropna_a2 = df_dropna[ (df_dropna['exp_label']=='cohort a') & \
                            (df_dropna['location']!='middle')]
        df_dropna_a3 = df_dropna[ (df_dropna['exp_label']=='cohort a') & \
                            (df_dropna['location']!='tip')]
        df_dropna_b1 = df_dropna[ (df_dropna['exp_label']=='cohort b') & \
                            (df_dropna['location']!='base')]
        df_dropna_b2 = df_dropna[ (df_dropna['exp_label']=='cohort b') & \
                            (df_dropna['location']!='middle')]
        df_dropna_b3 = df_dropna[ (df_dropna['exp_label']=='cohort b') & \
                            (df_dropna['location']!='tip')]
        exp_labels = df['exp_label'].unique()


        # get df_tip_lifetime, df_base_lifetime, df_middle_lifetime
        df_tip_lifetime = df[ df['tip_lifetime']!=0 ]
        df_middle_lifetime = df[ df['middle_lifetime']!=0 ]
        df_base_lifetime = df[ df['base_lifetime']!=0 ]





    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(17, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.89, 1, 0.11])
    fig2 = left_page.inset_axes([0, 0.56, 1, 0.33])
    fig3 = left_page.inset_axes([0, 0.23, 1, 0.33])
    fig4 = right_page.inset_axes([0, 0.56, 1, 0.33])
    fig5 = right_page.inset_axes([0, 0.23, 1, 0.33])

    fig1_1 = fig1.inset_axes([0.13, 0.35, 0.2, 0.6])
    fig1_2 = fig1.inset_axes([0.45, 0.35, 0.2, 0.6])
    fig1_3 = fig1.inset_axes([0.77, 0.35, 0.2, 0.6])

    fig2_1 = fig2.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig2_2 = fig2.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig2_3 = fig2.inset_axes([0.6, 0.48, 0.3, 0.2])
    fig2_4 = fig2.inset_axes([0.13, 0.18, 0.3, 0.2])
    fig2_5 = fig2.inset_axes([0.6, 0.18, 0.3, 0.2])

    fig3_1 = fig3.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig3_2 = fig3.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig3_3 = fig3.inset_axes([0.6, 0.48, 0.3, 0.2])
    fig3_4 = fig3.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig3_5 = fig3.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig3_6 = fig3.inset_axes([0.77, 0.18, 0.2, 0.2])

    fig4_1 = fig4.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig4_2 = fig4.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig4_3 = fig4.inset_axes([0.6, 0.48, 0.3, 0.2])
    fig4_4 = fig4.inset_axes([0.13, 0.18, 0.3, 0.2])
    fig4_5 = fig4.inset_axes([0.6, 0.18, 0.3, 0.2])

    fig5_1 = fig5.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig5_2 = fig5.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig5_3 = fig5.inset_axes([0.6, 0.48, 0.3, 0.2])
    fig5_4 = fig5.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig5_5 = fig5.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig5_6 = fig5.inset_axes([0.77, 0.18, 0.2, 0.2])








    # color_list = [plt.cm.get_cmap('Pastel1')(0),
    #                 plt.cm.get_cmap('Pastel1')(1),
    #                 plt.cm.get_cmap('Pastel1')(2),
    #                 ]

    color_list = [plt.cm.get_cmap('coolwarm')(0),
                    plt.cm.get_cmap('coolwarm')(0.7),
                    plt.cm.get_cmap('coolwarm')(0.99),
                    ]


    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1, fig2, fig3, fig4]:
        axis.set_xticks([]); axis.set_yticks([])

    # """
	# ~~~~Plot fig1~~~~
	# """
    # Plot fig1_1: mean msd curve (cohort a vs cohort b)
    add_mean_msd(fig1_1, df,
                cat_col='exp_label',
                cat_order=['cohort b', 'cohort a'],
                pixel_size=0.04,
                frame_rate=0.46,
                divide_num=2,
                RGBA_alpha=0.3,
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
                xscale=[-2, 34, 10],
                yscale=[-10000, 300000, 100000],
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

    # Plot fig1_2: D (cohort a vs cohort b)
    add_D_hist(fig1_2,
                df=df_particle,
                cat_col='exp_label',
                cat_order=['cohort b', 'cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                set_format=False,
                )

    add_t_test(fig1_2,
                blobs_df=df_particle,
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

    # Plot fig1_3: alpha (cohort a vs cohort b)
    add_alpha_hist(fig1_3,
                df=df_particle,
                cat_col='exp_label',
                cat_order=['cohort b', 'cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                set_format=False,
                )

    add_t_test(fig1_3,
                blobs_df=df_particle,
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
                yscale=[0, 15, 5],
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

    fig1.text(-0.05,
            -0.05,
            """
            Fig 1. Global MSD comparison between cohort a and cohort b.
            """,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize = 9,
            color = (0, 0, 0),
            transform=fig1.transAxes,
            weight = 'normal',
            fontname = 'Arial')


    # """
	# ~~~~Plot fig2~~~~
	# """
    # Plot fig2_1: v_norm vs direction
    add_hist(fig2_1,
                df=df_dropna,
                data_col='v_norm_abs',
                cat_col='direction',
                cat_order=None,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig2_1,
                blobs_df=df_dropna,
                cat_col='direction',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig2_1,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.5, 0.1],
                yscale=[0, 200, 50],
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
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles, labels = fig2_1.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['All', labels[0], labels[1]]
    fig2_1.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})

    # Plot fig2_2: cohort a v_norm vs direction
    add_hist(fig2_2,
                df=df_dropna[ df_dropna['exp_label']=='cohort a' ],
                data_col='v_norm_abs',
                cat_col='direction',
                cat_order=None,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig2_2,
                blobs_df=df_dropna[ df_dropna['exp_label']=='cohort a' ],
                cat_col='direction',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig2_2,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.5, 0.1],
                yscale=[0, 120, 50],
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
    handles, labels = fig2_2.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['cohort a', labels[0], labels[1]]
    fig2_2.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})

    # Plot fig2_3: cohort b v_norm vs direction
    add_hist(fig2_3,
                df=df_dropna[ df_dropna['exp_label']=='cohort b' ],
                data_col='v_norm_abs',
                cat_col='direction',
                cat_order=None,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig2_3,
                blobs_df=df_dropna[ df_dropna['exp_label']=='cohort b' ],
                cat_col='direction',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig2_3,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.5, 0.1],
                yscale=[0, 120, 50],
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
    handles, labels = fig2_3.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['cohort b', labels[0], labels[1]]
    fig2_3.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})

    # Plot fig2_4: toward to tip: v_norm_abs vs exp_label
    add_hist(fig2_4,
                df=df_dropna[ df_dropna['direction']=='toward to tip' ],
                data_col='v_norm_abs',
                cat_col='exp_label',
                cat_order=['cohort b', 'cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig2_4,
                blobs_df=df_dropna[ df_dropna['direction']=='toward to tip' ],
                cat_col='exp_label',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig2_4,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.5, 0.1],
                yscale=[0, 120, 50],
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
    handles, labels = fig2_4.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['toward to tip', labels[0], labels[1]]
    fig2_4.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})

    # Plot fig2_5: toward to tip: v_norm_abs vs exp_label
    add_hist(fig2_5,
                df=df_dropna[ df_dropna['direction']=='toward to base' ],
                data_col='v_norm_abs',
                cat_col='exp_label',
                cat_order=['cohort b', 'cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig2_5,
                blobs_df=df_dropna[ df_dropna['direction']=='toward to base' ],
                cat_col='exp_label',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.3],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6
                )
    format_ax(fig2_5,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.5, 0.1],
                yscale=[0, 120, 50],
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
    handles, labels = fig2_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['toward to base', labels[0], labels[1]]
    fig2_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})

    fig2.text(-0.05,
            0,
            """
            Fig 2. Speed vs Direction.
            """,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize = 9,
            color = (0, 0, 0),
            transform=fig2.transAxes,
            weight = 'normal',
            fontname = 'Arial')

    # """
	# ~~~~Plot fig3~~~~
	# """
    # Plot fig3_1: v_norm_abs vs location
    sns.lineplot(x="dist_to_base", y="v_norm_abs", hue="exp_label",
                hue_order=['cohort b', 'cohort a'],
                palette='coolwarm', data=df_dropna, ax=fig3_1)
    format_ax(fig3_1,
                xlabel='Dist to base (rel_length)',
                ylabel='Speed (rel_length/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-0.1, 1.1, 0.5],
                yscale=[-0.05, 0.22, 0.1],
                label_fontname='Liberation Sans',
                label_fontweight=9,
                label_fontsize=9,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=8,
                tklabel_fontsize=8,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # Plot fig3_2: cohort a v_norm vs location
    add_hist(fig3_2,
                df=df_dropna[ df_dropna['exp_label']=='cohort a' ],
                data_col='v_norm_abs',
                cat_col='location',
                cat_order=['base', 'middle', 'tip'],
                color_list=color_list,
                RGBA_alpha=1,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig3_2,
                blobs_df=df_dropna_a1,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.9],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='tip vs middle: ',
                horizontalalignment='left',
                )
    add_t_test(fig3_2,
                blobs_df=df_dropna_a2,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.82],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='tip vs base: ',
                horizontalalignment='left',
                )
    add_t_test(fig3_2,
                blobs_df=df_dropna_a3,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.74],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='middle vs base: ',
                horizontalalignment='left',
                )
    format_ax(fig3_2,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.75, 0.25],
                # yscale=[0, 8, 4],
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
    handles, labels = fig3_2.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1], handles[2]]
    labels = ['cohort a', labels[0], labels[1], labels[2]]
    fig3_2.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})
    # fig3_2.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig3_3: cohort b v_norm vs location
    add_hist(fig3_3,
                df=df_dropna[ df_dropna['exp_label']=='cohort b' ],
                data_col='v_norm_abs',
                cat_col='location',
                cat_order=['base', 'middle', 'tip'],
                color_list=color_list,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig3_3,
                blobs_df=df_dropna_b1,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.9],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='tip vs middle: ',
                horizontalalignment='left',
                )
    add_t_test(fig3_3,
                blobs_df=df_dropna_b2,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.82],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='tip vs base: ',
                horizontalalignment='left',
                )
    add_t_test(fig3_3,
                blobs_df=df_dropna_b3,
                cat_col='location',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.1, 0.74],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='middle vs base: ',
                horizontalalignment='left',
                )
    format_ax(fig3_3,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.75, 0.25],
                # yscale=[0, 8, 4],
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
    handles, labels = fig3_3.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1], handles[2]]
    labels = ['cohort b',
                labels[0],
                labels[1],
                labels[2],
                ]
    fig3_3.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})
    # fig3_3.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig3_4: cohort a v_norm vs location
    add_hist(fig3_4,
                df=df_dropna[ df_dropna['location']=='tip' ],
                data_col='v_norm_abs',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig3_4,
                blobs_df=df_dropna[ df_dropna['location']=='tip' ],
                cat_col='exp_label',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig3_4,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.75, 0.25],
                yscale=[0, 50, 10],
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
    handles, labels = fig3_4.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['tip', labels[0], labels[1]]
    fig3_4.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})
    # fig3_4.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig3_5: cohort a v_norm vs location
    add_hist(fig3_5,
                df=df_dropna[ df_dropna['location']=='middle' ],
                data_col='v_norm_abs',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig3_5,
                blobs_df=df_dropna[ df_dropna['location']=='middle' ],
                cat_col='exp_label',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig3_5,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.75, 0.25],
                yscale=[0, 150, 50],
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
    handles, labels = fig3_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['middle', labels[0], labels[1]]
    fig3_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})
    # fig3_5.set_yticklabels([-0.5, 0, 0.5, 1])

    # Plot fig3_6: cohort a v_norm vs location
    add_hist(fig3_6,
                df=df_dropna[ df_dropna['location']=='base' ],
                data_col='v_norm_abs',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig3_6,
                blobs_df=df_dropna[ df_dropna['location']=='base' ],
                cat_col='exp_label',
                hist_col='v_norm_abs',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig3_6,
                xlabel='Speed (relative_length / s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 0.75, 0.25],
                yscale=[0, 50, 10],
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
    handles, labels = fig3_6.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['base', labels[0], labels[1]]
    fig3_6.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})
    # fig3_6.set_yticklabels([-0.5, 0, 0.5, 1])

    fig3.text(-0.05,
            0,
            """
            Fig 3. Speed vs Location.
            """,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize = 9,
            color = (0, 0, 0),
            transform=fig3.transAxes,
            weight = 'normal',
            fontname = 'Arial')




    # """
	# ~~~~Plot fig5~~~~
	# """
    # Plot fig5_4:
    add_hist(fig5_4,
                df=df_tip_lifetime,
                data_col='tip_lifetime',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                bins=5,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig5_4,
                blobs_df=df_tip_lifetime,
                cat_col='exp_label',
                hist_col='tip_lifetime',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig5_4,
                xlabel='Tip lifetime (s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 15, 5],
                yscale=[0, 100, 50],
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
    handles, labels = fig5_4.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['tip_lifetime',
                labels[0] + ' (' + str(len(df_tip_lifetime[ df_tip_lifetime['exp_label']=='cohort b' ])) + ')',
                labels[1] + ' (' + str(len(df_tip_lifetime[ df_tip_lifetime['exp_label']=='cohort a' ])) + ')',
                ]
    fig5_4.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})


    # Plot fig5_5
    add_hist(fig5_5,
                df=df_middle_lifetime,
                data_col='middle_lifetime',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig5_5,
                blobs_df=df_middle_lifetime,
                cat_col='exp_label',
                hist_col='middle_lifetime',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig5_5,
                xlabel='Middle lifetime (s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 50, 10],
                yscale=[0, 120, 50],
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
    handles, labels = fig5_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['middle_lifetime',
                labels[0] + ' (' + str(len(df_middle_lifetime[ df_middle_lifetime['exp_label']=='cohort b' ])) + ')',
                labels[1] + ' (' + str(len(df_middle_lifetime[ df_middle_lifetime['exp_label']=='cohort a' ])) + ')',
                ]
    fig5_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})


    # # Plot fig5_6:
    add_hist(fig5_6,
                df=df_base_lifetime,
                data_col='base_lifetime',
                cat_col='exp_label',
                cat_order=['cohort b','cohort a'],
                bins=10,
                hist_kws={'linewidth':0.5},
                kde=False,
                )

    add_t_test(fig5_6,
                blobs_df=df_base_lifetime,
                cat_col='exp_label',
                hist_col='base_lifetime',
                drop_duplicates=False,
                text_pos=[0.9, 0.2],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=6,
                fontsize=6,
                prefix_str='',
                horizontalalignment='right',
                )
    format_ax(fig5_6,
                xlabel='Base lifetime (s)',
                ylabel='Counts',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 15, 5],
                yscale=[0, 100, 50],
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
    handles, labels = fig5_6.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['base_lifetime',
                labels[0] + ' (' + str(len(df_base_lifetime[ df_base_lifetime['exp_label']=='cohort b' ])) + ')',
                labels[1] + ' (' + str(len(df_base_lifetime[ df_base_lifetime['exp_label']=='cohort a' ])) + ')',
                ]
    fig5_6.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 6, 'weight' : 6})



    fig5.text(-0.05,
            0,
            """
            Fig 3. Lifetime vs Location.
            """,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize = 9,
            color = (0, 0, 0),
            transform=fig5.transAxes,
            weight = 'normal',
            fontname = 'Arial')
































































    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    # plt.clf(); plt.close()
