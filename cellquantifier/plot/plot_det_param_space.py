import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .plotutil import *

def plot_det_param_space(
    df, frame,
    blob_markersize=3,
    plot_r=False,
    ):
    """
    Plot detections in parameter spaces.

    Pseudo code
    ----------
    1. Format figures into standard format.
    2. Add scatters in the ax.

    Parameters
    ----------
    df : DataFrame
		DataFrame containing necessary data.

    frame : 2darray
		Frame with the format of 2D Numpy array.

    Returns
    -------
    A matplotlib Figure instance.

    Examples
	--------
    """

    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 4
    row_num = 1
    divide_index = []
    hidden_index = []
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 2
    index_s1 = [
        ]
    # Sub_axs_2 settings
    col_num_s2 = 1
    row_num_s2 = 2
    index_s2 = [
        ]

    # """
	# ~~~~Initialize the colors~~~~
	# """
    print("\n")
    print("Preparing colors")
    palette = sns.color_palette('muted', col_num*row_num)
    # c1 = palette[0]
    # c2 = palette[1]
    # c3 = palette[2]
    # RGBA_alpha = 0.8


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
	# ~~~~Prepare df for the whole page~~~~
	# """
    print("\n")
    print("Preparing data")


    # """
	# ~~~~Plot parameter spaces~~~~
	# """
    figs = [
            axs[1], axs[2]
            ]
    datas = [
            df, df,
            ]
    param_cols = [
            ['r', 'peak'], ['r', 'mass'],
            ]
    data_cols = [
            None, None,
            ]
    colors = palette
    xlabels = [
            'Radius (px)', 'Radius (px)',
            ]
    ylabels = [
            'Peak (ADU)', 'Mass (ADU)'
            ]
    for i, (fig, data, param_col, data_col, color, xlabel, ylabel, ) \
    in enumerate(zip(figs, datas, param_cols, data_cols, colors, xlabels, ylabels, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        add_scatter_2d(ax=fig, df=data,
            axis_cols=param_col,
            data_col=data_col,
            color=color,
            )
        set_xylabel(fig,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    )


    # """
	# ~~~~Plot parameter spaces~~~~
	# """
    figs = [
            axs[0], axs[3]
            ]
    datas = [
            df, df,
            ]
    frames = [
            frame, frame,
            ]
    for i, (fig, data, frame) \
    in enumerate(zip(figs, datas, frames, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        fig.imshow(frame, cmap='gray', aspect='equal')
        fig.set_xlim(0, frame.shape[1])
        fig.set_ylim(0, frame.shape[0])
        for spine in ['top', 'bottom', 'left', 'right']:
            fig.spines[spine].set_visible(False)

        anno_blob(fig, data, marker='^',
                    markersize=blob_markersize,
                    plot_r=plot_r,
                    color=(0,0,1))


    # # """
	# # ~~~~Add figure text~~~~
	# # """
    # figs = [
    #         grids[0],
    #         ]
    # fig_texts = [
    #         'Fig.1. Foci_num vs frame',
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
    #
    # # """
	# # ~~~~Additional figures format~~~~
	# # """
    # # Format scale
    # figs = axs
    # xscales = [
    #         [0, 300, 50],
    #         ]
    # yscales = [
    #         [None, None],
    #         ]
    # for i, (fig, xscale, yscale, ) \
    # in enumerate(zip(figs, xscales, yscales,)):
    #     format_scale(fig,
    #             xscale=xscale,
    #             yscale=yscale,
    #             )


    return all_figures
