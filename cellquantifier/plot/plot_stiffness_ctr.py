import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2
from ..phys import *

def plot_stiffness_ctr(
    df=pd.DataFrame([]),
    ):

    """
    Plot a quick overview page of stiffness data based on "mergedPhysData" only.

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
    print("\n")
    print("Preparing data")
    if not df.empty:
        # """
    	# ~~~~Iterate different exp_label, calculate avg_foci_num~~~~
    	# """
        exp_labels = df['exp_label'].unique()
        for exp_label in exp_labels:
            df_exp = df[ df['exp_label']==exp_label ]

            frames = df_exp['frame'].unique()
            for frame in frames:
                curr_df = df_exp[ df_exp['frame']==frame ]

                avg_foci_num = curr_df['foci_num'].mean()
                df.loc[ (df['exp_label']==exp_label)&(df['frame']==frame), 'avg_foci_num' ] = avg_foci_num

        # """
        # ~~~~Divide df into sub_dfs~~~~
        # """
        dfp1 = df[ df['exp_label']=='Ctrl-24h-glass' ].drop_duplicates('frame')
        dfp1 = dfp1[ ['frame', 'avg_foci_num', 'exp_label'] ]
        dfp1 = dfp1.sort_values(by='frame')
        dfp1.round(3).to_csv('/home/linhua/Desktop/stiffness_ctrl.csv', index=False)


	# ~~~~Initialize the colors~~~~
	# """
    # Layout settings
    col_num = 1
    row_num = 1
    divide_index = []
    hidden_index = []

    print("\n")
    print("Preparing colors")
    palette = sns.cubehelix_palette(4, start=2, rot=0, dark=0, light=.5, reverse=True)
    # palette = sns.color_palette('muted', col_num*row_num)
    c4 = palette[0]
    c3 = palette[1]
    c2 = palette[2]
    c1 = palette[3]
    c0 = [0.3, 0.3, 0.3]

    p = palette
    RGBA_alpha = 0.9
    c1 = (c1[0], c1[1], c1[2], RGBA_alpha)
    c2 = (c2[0], c2[1], c2[2], RGBA_alpha)
    c3 = (c3[0], c3[1], c3[2], RGBA_alpha)
    c4 = (c4[0], c4[1], c4[2], RGBA_alpha)


    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 3
    index_s1 = [
        ]
    # Sub_axs_2 settings
    col_num_s2 = 1
    row_num_s2 = 2
    index_s2 = [
        ]

    # Layout implementation
    print("\n")
    print("Preparing layout")
    tot_width = col_num * 4
    tot_height = row_num * 3
    all_figures, page = plt.subplots(1, 1, figsize=(tot_width, tot_height))

    grids = []
    axs = []

    axs_s1_bg = []
    axs_s1 = []
    axs_s1_base = []
    axs_s1_slave = []

    axs_s2_bg = []
    axs_s2 = []
    axs_s2_base = []
    axs_s2_slave = []
    for i in range(col_num*row_num):
        r = i // col_num
        c = i % col_num
        w = 1 / col_num
        h = 1 / row_num
        x0 = c * w
        y0 = 1 - (r+1) * h

        # Generate Grids
        grids.append(page.inset_axes([x0, y0, w, h]))

        # Generate individual axs
        axs.append(grids[i].inset_axes([0.33, 0.33, 0.6, 0.6]))

        # Customize axs_s1
        if i in index_s1:
            axs_s1_bg.append(axs[i])
            for i_s1 in range(col_num_s1*row_num_s1):
                r_s1 = i_s1 // col_num_s1
                c_s1 = i_s1 % col_num_s1
                w_s1 = 1 / col_num_s1
                h_s1 = 1 / row_num_s1
                x0_s1 = c_s1 * w_s1
                y0_s1 = 1 - (r_s1+1) * h_s1
                # Generate axs_s1, axs_s1_base, axs_s1_slave
                temp = axs[i].inset_axes([x0_s1, y0_s1, w_s1, h_s1])
                axs_s1.append(temp)
                if y0_s1 == 0:
                    axs_s1_base.append(temp)
                else:
                    axs_s1_slave.append(temp)

        # Customize axs_s2
        if i in index_s2:
            axs_s2_bg.append(axs[i])
            for i_s2 in range(col_num_s2*row_num_s2):
                r_s2 = i_s2 // col_num_s2
                c_s2 = i_s2 % col_num_s2
                w_s2 = 1 / col_num_s2
                h_s2 = 1 / row_num_s2
                x0_s2 = c_s2 * w_s2
                y0_s2 = 1 - (r_s2+1) * h_s2
                # Generate axs_s2, axs_s2_base, axs_s2_slave
                temp = axs[i].inset_axes([x0_s2, y0_s2, w_s2, h_s2])
                axs_s2.append(temp)
                if y0_s2 == 0:
                    axs_s2_base.append(temp)
                else:
                    axs_s2_slave.append(temp)

    # """
	# ~~~~format figures~~~~
	# """
    print("\n")
    print("Formating figures")
    # Format page
    for ax in [page]:
        ax.set_xticks([]);
        ax.set_yticks([])
        format_spine(ax, spine_linewidth=2)

    # Format grids
    for ax in grids:
        ax.set_xticks([]);
        ax.set_yticks([])
        format_spine(ax, spine_linewidth=2)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    for i in divide_index:
        for spine in ['bottom']:
            grids[i].spines[spine].set_visible(True)

    # Format axs
    for ax in axs:
        format_spine(ax, spine_linewidth=0.5)
        format_tick(ax, tk_width=0.5)
        format_tklabel(ax, tklabel_fontsize=10)
        format_label(ax, label_fontsize=10)

    for i in hidden_index:
        axs[i].set_xticks([]);
        axs[i].set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            axs[i].spines[spine].set_visible(False)

    # Format sub_axs_background
    for ax in axs_s1_bg + axs_s2_bg:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    # Format sub_axs
    for ax in axs_s1 + axs_s2:
        format_spine(ax, spine_linewidth=0.5)
        format_tick(ax, tk_width=0.5)
        format_tklabel(ax, tklabel_fontsize=10)
        format_label(ax, label_fontsize=10)
        ax.set_yticks([])

    # Format sub_axs_slave
    for ax in axs_s1_slave + axs_s2_slave:
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # empty_string_labels = ['']*len(labels)
        # ax.set_xticklabels(empty_string_labels)
        #
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # empty_string_labels = ['']*len(labels)
        # ax.set_yticklabels(empty_string_labels)
        ax.set_xticks([])


    # """
	# ~~~~Plot foci_num~~~~
	# """
    figs = [
            axs[0],
            ]
    datas = [
            dfp1,
            ]
    x_cols = [
            'frame',
            ]
    y_cols = [
            'avg_foci_num',
            ]
    colors = [
            c0,
            ]
    legends = [
            'Ctr-24h-glass',
            ]
    xlabels = [
            'frame',
            ]
    ylabels = [
            'average foci num',
            ]
    for i, (fig, data, x_col, y_col, color, legend, xlabel, ylabel) \
    in enumerate(zip(figs, datas, x_cols, y_cols, colors, legends, xlabels, ylabels)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        fig.plot(data[x_col], data[y_col], color=color, label=legend)

        format_legend(fig,
                show_legend=True,
                legend_loc='lower right',
                legend_fontweight=5,
                legend_fontsize=5,
                )

        set_xylabel(fig,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    )

    # # """
	# # ~~~~Add figure text~~~~
	# # """
    # figs = grids
    # fig_texts = [
    #         'Fig.1a. Mean MSD curve comparion',
    #         'Fig.1b. D value comparison',
    #         'Fig.1c. Alpha value comparion',
    #         ]
    # for i, (fig, fig_text, ) \
    # in enumerate(zip(figs, fig_texts, )):
    #     print("\n")
    #     print("Plotting (%d/%d)" % (i+1, len(figs)))
    #
    #     fig.text(0.1,
    #             0.05,
    #             fig_text,
    #             horizontalalignment='left',
    #             color=(0,0,0,1),
    #             family='Liberation Sans',
    #             fontweight=10,
    #             fontsize=10,
    #             transform=fig.transAxes,
    #             )


    # """
	# ~~~~Additional figures format~~~~
	# """
    # Format scale
    figs = [
            axs[0],
            ]
    xscales = [
            [-10, 300],
            ]
    yscales = [
            [None, None],
            ]
    for i, (fig, xscale, yscale, ) \
    in enumerate(zip(figs, xscales, yscales,)):
        format_scale(fig,
                xscale=xscale,
                yscale=yscale,
                )


    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
	# """
    return all_figures
