import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .plotutil import *

def plot_foci_prop_hist(
    df,
    ):

    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 4
    row_num = 3
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
    c1 = palette[0]
    c2 = palette[1]
    c3 = palette[2]
    RGBA_alpha = 0.8


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
    if not df.empty:
        df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        cols = [
            'peak', 'mass', 'mean', 'std',
            'A', 'area', 'slope', 'dist_err',
            'sig_x', 'sig_y', 'sigx_to_sigraw', 'sigy_to_sigraw',
                ]
        for col in cols:
            df = df[ (df[col]<df[col].quantile(0.99)) & \
                    (df[col]>df[col].quantile(0.01))]


    # """
	# ~~~~Plot hist~~~~
	# """
    figs = axs
    datas = [
            df, df, df, df,
            df, df, df, df,
            df, df, df, df,
            ]
    bins = [
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            ]
    data_cols = [
            'peak', 'mass', 'mean', 'std',
            'A', 'area', 'slope', 'dist_err',
            'sig_x', 'sig_y', 'sigx_to_sigraw', 'sigy_to_sigraw',
            ]
    colors = palette
    for i, (fig, data, bin, data_col, color, ) \
    in enumerate(zip(figs, datas, bins, data_cols, colors, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        sns.distplot(data[data_col],
                    bins=bin,
                    kde=True,
                    color=color,
                    ax=fig,
                    hist_kws={"alpha": RGBA_alpha,
                    'linewidth': 0.5, 'edgecolor': (0,0,0)},
                    kde_kws={"alpha": RGBA_alpha,
                    'linewidth': 1.5, 'color': (0,0,0)},
                    )


    return all_figures
