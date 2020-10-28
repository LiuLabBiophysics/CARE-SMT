import pandas as pd; import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from ..phys.physutil import *
from ..plot.plotutil import *


def fig_quick_nucleosome(df=pd.DataFrame([])):
    """
    Plot a quick overview page of phys data based on "mergedPhysData" only.

    Pseudo code
    ----------
    1. Prepare df for the whole page.
    2. Initialize the page layout.

    Parameters
    ----------
    df : DataFrame, optional
        mergedPhysData to be plotted.

    Returns
    -------
    A pdf page of plots based on "mergedPhysData".

    Examples
	--------

    """
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty:
        df.loc[ df['sort_flag_boundary']==0, 'sort_flag_boundary'] = 'Interior'
        df.loc[ df['sort_flag_boundary']==1, 'sort_flag_boundary' ] = 'Boundary'
        df.loc[ df['sort_flag_53bp1']==0, 'sort_flag_53bp1'] = 'Non-53BP1'
        df.loc[ df['sort_flag_53bp1']==1, 'sort_flag_53bp1' ] = '53BP1'

        # filter out traj_length_thres
        df = df [ df['traj_length']>80 ]

        # add 'avg_dist_bound', 'avg_dist_53bp1' to df
        # Set dist to bound bin_size=3
        # Set dist to 53bp1 bin_size=1
        df = add_avg_dist(df)
        df['avg_dist_bound'] = (df['avg_dist_bound'] // 3) * 3
        df['avg_dist_53bp1'] = round(df['avg_dist_53bp1'])

        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')
        exp_labels = df['exp_label'].unique()


    # """
	# ~~~~Try~~~~
	# """
    # sns.lineplot(x="avg_dist_53bp1", y="D", hue="exp_label",
    #             palette='coolwarm', data=df_particle)
    # plt.show()


    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(17, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.6666, 1, 0.3333])
    fig2 = left_page.inset_axes([0, 0.3333, 1, 0.3333])
    fig3 = left_page.inset_axes([0, 0, 1, 0.3333])
    # fig5 = right_page.inset_axes([0, 0.75, 1, 0.25])
    # fig6 = right_page.inset_axes([0, 0.5, 1, 0.25])
    # fig7 = right_page.inset_axes([0, 0.25, 1, 0.25])
    # fig8 = right_page.inset_axes([0, 0, 1, 0.25])

    fig1_1 = fig1.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig1_2 = fig1.inset_axes([0.6, 0.78, 0.3, 0.2])
    fig1_3 = fig1.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig1_4 = fig1.inset_axes([0.6, 0.48, 0.1, 0.2])
    fig1_5 = fig1.inset_axes([0.8, 0.48, 0.1, 0.2])
    fig1_6 = fig1.inset_axes([0.13, 0.18, 0.3, 0.2])
    fig1_7 = fig1.inset_axes([0.6, 0.18, 0.1, 0.2])
    fig1_8 = fig1.inset_axes([0.8, 0.18, 0.1, 0.2])

    fig2_1 = fig2.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig2_2 = fig2.inset_axes([0.6, 0.78, 0.3, 0.2])
    fig2_3 = fig2.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig2_4 = fig2.inset_axes([0.6, 0.48, 0.1, 0.2])
    fig2_5 = fig2.inset_axes([0.8, 0.48, 0.1, 0.2])
    fig2_6 = fig2.inset_axes([0.13, 0.18, 0.3, 0.2])
    fig2_7 = fig2.inset_axes([0.6, 0.18, 0.1, 0.2])
    fig2_8 = fig2.inset_axes([0.8, 0.18, 0.1, 0.2])

    fig3_1 = fig3.inset_axes([0.13, 0.78, 0.3, 0.2])
    fig3_2 = fig3.inset_axes([0.6, 0.78, 0.1, 0.2])
    fig3_3 = fig3.inset_axes([0.8, 0.78, 0.1, 0.2])
    fig3_4 = fig3.inset_axes([0.13, 0.48, 0.3, 0.2])
    fig3_5 = fig3.inset_axes([0.6, 0.48, 0.1, 0.2])
    fig3_6 = fig3.inset_axes([0.8, 0.48, 0.1, 0.2])
    fig3_7 = fig3.inset_axes([0.13, 0.18, 0.3, 0.2])
    fig3_8 = fig3.inset_axes([0.6, 0.18, 0.1, 0.2])
    fig3_9 = fig3.inset_axes([0.8, 0.18, 0.1, 0.2])

    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1, fig2, fig3]:
        axis.set_xticks([]); axis.set_yticks([])


    # """
	# ~~~~Prepare fig1 data; Plot fig1~~~~
	# """
    # Prepare df_fig1
    df_fig1 = df_particle.copy()
    df_fig1 = df_fig1[ df_fig1['avg_dist_bound']>-60 ]
    df_fig1 = df_fig1[ df_fig1['avg_dist_bound']<=0 ]
    df_fig1['avg_dist_bound'] = df_fig1['avg_dist_bound'] * 0.163

    # # Plot fig1_1: avg_dist_bound vs D
    sns.lineplot(x="avg_dist_bound", y="D", hue="exp_label",
                palette='coolwarm', data=df_fig1, ax=fig1_1)

    format_ax(fig1_1,
                xlabel='Distance to boundary (um)',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-10, 0.5, 2],
                yscale=[2000, 10000, 2000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                )

    handles, labels = fig1_1.get_legend_handles_labels()
    fig1_1.legend(handles=handles[1:], labels=labels[1:],
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig1_2: avg_dist_bound vs alpha
    sns.lineplot(x="avg_dist_bound", y="alpha", hue="exp_label",
                palette='coolwarm', data=df_fig1, ax=fig1_2)

    format_ax(fig1_2,
                xlabel='Distance to boundary (um)',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-10, 0.5, 2],
                yscale=[0, 0.5, 0.1],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                )

    handles, labels = fig1_2.get_legend_handles_labels()
    fig1_2.legend(handles=handles[1:], labels=labels[1:],
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig1_3: exp_labels[0] mean msd curve (Interior vs Boundary)
    add_mean_msd(fig1_3, df[ df['exp_label']==exp_labels[0]],
                cat_col='sort_flag_boundary',
                cat_order=['Interior', 'Boundary'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig1_3,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig1_3.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig1_3.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig1_4: exp_labels[0] D (Interior vs Boundary)
    add_violin_2(fig1_4,
                df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                data_col='D',
                cat_col='sort_flag_boundary',
                hue_order=['Interior', 'Boundary'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig1_4,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                cat_col='sort_flag_boundary',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig1_4,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig1_4.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig1_4.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig1_4.get_xaxis().set_ticks([])


    # Plot fig1_5: exp_labels[0] alpha (Interior vs Boundary)
    add_violin_2(fig1_5,
                df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                data_col='alpha',
                cat_col='sort_flag_boundary',
                hue_order=['Interior', 'Boundary'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig1_5,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                cat_col='sort_flag_boundary',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig1_5,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig1_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig1_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig1_5.get_xaxis().set_ticks([])


    # Plot fig1_6: exp_labels[1] mean msd curve (Interior vs Boundary)
    add_mean_msd(fig1_6, df[ df['exp_label']==exp_labels[1]],
                cat_col='sort_flag_boundary',
                cat_order=['Interior', 'Boundary'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig1_6,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig1_6.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig1_6.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})


    # Plot fig1_7: exp_labels[1] D (Interior vs Boundary)
    add_violin_2(fig1_7,
                df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                data_col='D',
                cat_col='sort_flag_boundary',
                hue_order=['Interior', 'Boundary'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig1_7,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                cat_col='sort_flag_boundary',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig1_7,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig1_7.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig1_7.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig1_7.get_xaxis().set_ticks([])


    # Plot fig1_8: exp_labels[1] alpha (Interior vs Boundary)
    add_violin_2(fig1_8,
                df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                data_col='alpha',
                cat_col='sort_flag_boundary',
                hue_order=['Interior', 'Boundary'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig1_8,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                cat_col='sort_flag_boundary',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig1_8,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig1_8.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig1_8.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig1_8.get_xaxis().set_ticks([])



    # """
	# ~~~~Prepare fig2 data; Plot fig2~~~~
	# """
    # Prepare df_fig2
    df_fig2 = df_particle.copy()
    df_fig2 = df_fig2[ df_fig2['avg_dist_53bp1']<20 ]
    df_fig2['avg_dist_53bp1'] = df_fig2['avg_dist_53bp1'] * 0.163

    # # Plot fig2_1: avg_dist_53bp1 vs D
    sns.lineplot(x="avg_dist_53bp1", y="D", hue="exp_label",
                palette='coolwarm', data=df_fig2, ax=fig2_1)

    format_ax(fig2_1,
                xlabel='Distance to 53bp1 (um)',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-0.5, 3.5, 1],
                yscale=[0, 9000, 2000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                )

    handles, labels = fig2_1.get_legend_handles_labels()
    fig2_1.legend(handles=handles[1:], labels=labels[1:],
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig2_2: avg_dist_53bp1 vs alpha
    sns.lineplot(x="avg_dist_53bp1", y="alpha", hue="exp_label",
                palette='coolwarm', data=df_fig2, ax=fig2_2)

    format_ax(fig2_2,
                xlabel='Distance to 53bp1 (um)',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[-0.5, 3.5, 1],
                yscale=[-0.2, 0.5, 0.1],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                )

    handles, labels = fig2_2.get_legend_handles_labels()
    fig2_2.legend(handles=handles[1:], labels=labels[1:],
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig2_3: exp_labels[0] mean msd curve (Non-53BP1 vs 53BP1)
    add_mean_msd(fig2_3, df[ df['exp_label']==exp_labels[0]],
                cat_col='sort_flag_53bp1',
                cat_order=['Non-53BP1', '53BP1'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig2_3,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig2_3.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig2_3.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})

    # Plot fig2_4: exp_labels[0] D (Non-53BP1 vs 53BP1)
    add_violin_2(fig2_4,
                df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                data_col='D',
                cat_col='sort_flag_53bp1',
                hue_order=['Non-53BP1', '53BP1'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig2_4,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                cat_col='sort_flag_53bp1',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig2_4,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig2_4.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig2_4.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig2_4.get_xaxis().set_ticks([])


    # Plot fig2_5: exp_labels[0] alpha (Non-53BP1 vs 53BP1)
    add_violin_2(fig2_5,
                df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                data_col='alpha',
                cat_col='sort_flag_53bp1',
                hue_order=['Non-53BP1', '53BP1'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig2_5,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[0] ],
                cat_col='sort_flag_53bp1',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig2_5,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig2_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[0], labels[0], labels[1]]
    fig2_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig2_5.get_xaxis().set_ticks([])


    # Plot fig2_6: exp_labels[1] mean msd curve (Non-53BP1 vs 53BP1)
    add_mean_msd(fig2_6, df[ df['exp_label']==exp_labels[1]],
                cat_col='sort_flag_53bp1',
                cat_order=['Non-53BP1', '53BP1'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig2_6,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig2_6.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig2_6.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})


    # Plot fig2_7: exp_labels[1] D (Non-53BP1 vs 53BP1)
    add_violin_2(fig2_7,
                df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                data_col='D',
                cat_col='sort_flag_53bp1',
                hue_order=['Non-53BP1', '53BP1'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig2_7,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                cat_col='sort_flag_53bp1',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig2_7,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig2_7.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig2_7.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig2_7.get_xaxis().set_ticks([])


    # Plot fig2_8: exp_labels[1] alpha (Non-53BP1 vs 53BP1)
    add_violin_2(fig2_8,
                df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                data_col='alpha',
                cat_col='sort_flag_53bp1',
                hue_order=['Non-53BP1', '53BP1'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig2_8,
                blobs_df=df_particle[ df_particle['exp_label']==exp_labels[1] ],
                cat_col='sort_flag_53bp1',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig2_8,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig2_8.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = [exp_labels[1], labels[0], labels[1]]
    fig2_8.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig2_8.get_xaxis().set_ticks([])









    # """
	# ~~~~Prepare fig3 data; Plot fig3~~~~
	# """
    # Prepare df_fig3
    df_fig3 = df_particle.copy()


    # Plot fig3_1: 'All' mean msd curve (Living vs BLM)
    add_mean_msd(fig3_1, df,
                cat_col='exp_label',
                cat_order=['50NcLiving', '50NcBLM'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig3_1,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig3_1.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['All', labels[0], labels[1]]
    fig3_1.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})


    # Plot fig3_2: 'All' D (Living vs BLM)
    add_violin_2(fig3_2,
                df=df_particle,
                data_col='D',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_2,
                blobs_df=df_particle,
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_2,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_2.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['All', labels[0], labels[1]]
    fig3_2.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_2.get_xaxis().set_ticks([])


    # Plot fig3_3: 'All' alpha (Living vs BLM)
    add_violin_2(fig3_3,
                df=df_particle,
                data_col='alpha',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_3,
                blobs_df=df_particle,
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_3,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_3.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['All', labels[0], labels[1]]
    fig3_3.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_3.get_xaxis().set_ticks([])



    # Plot fig3_4: '53BP1' mean msd curve (Living vs BLM)
    add_mean_msd(fig3_4, df[df['sort_flag_53bp1']=='53BP1'],
                cat_col='exp_label',
                cat_order=['50NcLiving', '50NcBLM'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig3_4,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig3_4.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['53BP1', labels[0], labels[1]]
    fig3_4.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})


    # Plot fig3_5: '53BP1' D (Living vs BLM)
    add_violin_2(fig3_5,
                df=df_particle[df_particle['sort_flag_53bp1']=='53BP1'],
                data_col='D',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_5,
                blobs_df=df_particle[df_particle['sort_flag_53bp1']=='53BP1'],
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_5,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_5.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['53BP1', labels[0], labels[1]]
    fig3_5.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_5.get_xaxis().set_ticks([])


    # Plot fig3_6: '53BP1' alpha (Living vs BLM)
    add_violin_2(fig3_6,
                df=df_particle[df_particle['sort_flag_53bp1']=='53BP1'],
                data_col='alpha',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_6,
                blobs_df=df_particle[df_particle['sort_flag_53bp1']=='53BP1'],
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_6,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_6.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['53BP1', labels[0], labels[1]]
    fig3_6.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_6.get_xaxis().set_ticks([])


    # Plot fig3_7: 'Non-53BP1' mean msd curve (Living vs BLM)
    add_mean_msd(fig3_7, df[df['sort_flag_53bp1']=='Non-53BP1'],
                cat_col='exp_label',
                cat_order=['50NcLiving', '50NcBLM'],
                pixel_size=0.163,
                frame_rate=20,
                divide_num=5,
                RGBA_alpha=0.3,
                fitting_linewidth=1.5,
                elinewidth=0.5,
                markersize=4,
                capsize=1,
                set_format=False)

    format_ax(fig3_7,
                xlabel='Time (s)',
                ylabel=r'MSD (nm$^2$)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[0, 2.05, 0.5],
                yscale=[5000, 26000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=True,
                legend_loc='lower left',
                legend_frameon=False,
                legend_fontname='Liberation Sans',
                legend_fontweight=9,
                legend_fontsize=9,
                )

    handles, labels = fig3_7.get_legend_handles_labels()
    for i in range(len(labels)):
        if ':' in labels[i]:
            curr_label = labels[i]
            curr_label = curr_label[curr_label.find(':')+2:]
            labels[i] = curr_label
    empty_handle = mpl.patches.Rectangle((0,0), 1, 1, fill=False,
            edgecolor='none', visible=False)
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['Non-53BP1', labels[0], labels[1]]
    fig3_7.legend(handles=handles, labels=labels,
            loc='lower right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 9, 'weight' : 9})


    # Plot fig3_8: 'Non-53BP1' D (Living vs BLM)
    add_violin_2(fig3_8,
                df=df_particle[df_particle['sort_flag_53bp1']=='Non-53BP1'],
                data_col='D',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_8,
                blobs_df=df_particle[df_particle['sort_flag_53bp1']=='Non-53BP1'],
                cat_col='exp_label',
                hist_col='D',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_8,
                xlabel='',
                ylabel=r'D (nm$^2$/s)',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 15000, 5000],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_8.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['Non-53BP1', labels[0], labels[1]]
    fig3_8.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_8.get_xaxis().set_ticks([])


    # Plot fig3_9: 'Non-53BP1' alpha (Living vs BLM)
    add_violin_2(fig3_9,
                df=df_particle[df_particle['sort_flag_53bp1']=='Non-53BP1'],
                data_col='alpha',
                cat_col='exp_label',
                hue_order=['50NcLiving', '50NcBLM'],
                RGBA_alpha=0.3,
                )

    add_t_test(fig3_9,
                blobs_df=df_particle[df_particle['sort_flag_53bp1']=='Non-53BP1'],
                cat_col='exp_label',
                hist_col='alpha',
                drop_duplicates=False,
                text_pos=[0.99, 0.0],
                color=(0,0,0,1),
                fontname='Liberation Sans',
                fontweight=8,
                fontsize=8
                )

    format_ax(fig3_9,
                xlabel='',
                ylabel=r'$\alpha$',
                spine_linewidth=1,
                xlabel_color=(0,0,0,1),
                ylabel_color=(0,0,0,1),
                xscale=[],
                yscale=[0, 1, 0.5],
                label_fontname='Liberation Sans',
                label_fontweight=11,
                label_fontsize=11,
                tklabel_fontname='Liberation Sans',
                tklabel_fontweight=10,
                tklabel_fontsize=10,
                show_legend=False)
    handles, labels = fig3_9.get_legend_handles_labels()
    handles = [empty_handle, handles[0], handles[1]]
    labels = ['Non-53BP1', labels[0], labels[1]]
    fig3_9.legend(handles=handles, labels=labels,
            loc='upper right', frameon=False,
            prop={'family' : 'Liberation Sans', 'size' : 4, 'weight' : 4})
    fig3_9.get_xaxis().set_ticks([])
























    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    # plt.clf(); plt.close()
