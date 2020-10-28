import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2
from ..phys import *

def fig_quick_merge(
    df=pd.DataFrame([]),
    ):

    # """
	# ~~~~Initialize the colors~~~~
	# """
    print("\n")
    print("Preparing colors")
    c1 = (0, 0, 1)
    c2 = (0, 0, 0)
    c3 = (1, 0, 0)
    RGBA_alpha = 0.8
    p = [c1, c2, c3]


    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 3
    row_num = 1
    divide_index = []
    hidden_index = []
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 3
    index_s1 = [
        1, 2,
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
	# ~~~~Prepare df for the whole page~~~~
	# """
    print("\n")
    print("Preparing data")
    if not df.empty:
        # """
        # ~~~~Filters applied to df~~~~
        # """

        # traj_length filter
        if 'traj_length' in df:
            df = df[ df['traj_length']>=40 ]

        # """
        # ~~~~Divide df into sub_dfs~~~~
        # """
        dfp = df.drop_duplicates('particle')
        dfp_b = dfp[ dfp['exp_label']=='50NcLivingB' ]
        dfp_m = dfp[ dfp['exp_label']=='50NcLivingM' ]
        dfp_t = dfp[ dfp['exp_label']=='50NcLivingT' ]

        dfp_bm = dfp[ dfp['exp_label']!='50NcLivingT' ]
        dfp_mt = dfp[ dfp['exp_label']!='50NcLivingB' ]


    # """
	# ~~~~Plot msd~~~~
	# """
    rename_msd_figs = []
    msd_figs = [
        axs[0],
        ]
    datas = [
        df,
        ]
    palettes = [
        p,
        ]
    cat_cols = [
        'exp_label',
        ]
    orders = [
        ['50NcLivingB', '50NcLivingM', '50NcLivingT',],
        ]

    for i, (fig, data, palette, cat_col, order, ) \
    in enumerate(zip(msd_figs, datas, palettes, cat_cols, orders, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(msd_figs)))

        add_mean_msd2(fig, data,
                    pixel_size=0.163,
                    frame_rate=20,
                    divide_num=5,

                    cat_col=cat_col,
                    cat_order=order,
                    color_order=palette,
                    RGBA_alpha=RGBA_alpha,

                    fitting_linewidth=1,
                    elinewidth=1,
                    markersize=3,
                    capsize=1.5,
                    )
        set_xylabel(fig,
                    xlabel='Time (s)',
                    ylabel=r'MSD (nm$^2$)',
                    )


    # """
	# ~~~~Plot hist~~~~
	# """
    figs = axs_s1
    datas = [
            dfp_b, dfp_m, dfp_t,
            dfp_b, dfp_m, dfp_t,
            ]
    bins = [
            None, None, None,
            None, None, None,
            ]
    data_cols = [
            'D', 'D', 'D',
            'alpha', 'alpha', 'alpha',
            ]
    colors = [
            c1, c2, c3,
            c1, c2, c3,
            ]
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
                    'linewidth': 1.5, 'color': color},
                    )


    # """
	# ~~~~Add t test~~~~
	# """
    figs = [
            axs_s1[0], axs_s1[2],
            axs_s1[3], axs_s1[5],
            ]
    datas = [
            dfp_bm, dfp_mt,
            dfp_bm, dfp_mt,
            ]
    data_cols = [
            'D', 'D',
            'alpha', 'alpha',
            ]
    cat_cols = [
            'exp_label', 'exp_label',
            'exp_label', 'exp_label',
            ]
    text_poss = [
            # (0.98, 0.78), (0.98, 0.78),
            (0.98, 0.68), (0.98, 0.68),
            (0.98, 0.68), (0.98, 0.68),
            ]
    for i, (fig, data, data_col, cat_col, text_pos, ) \
    in enumerate(zip(figs, datas, data_cols, cat_cols, text_poss, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        add_t_test(fig,
                    blobs_df=data,
                    cat_col=cat_col,
                    hist_col=data_col,
                    drop_duplicates=False,
                    text_pos=text_pos,
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=9,
                    horizontalalignment='right',
                    format='general',
                    )
    # """
	# ~~~~Add figure text~~~~
	# """
    figs = grids
    fig_texts = [
            'Fig.1a. Mean MSD curve comparion',
            'Fig.1b. D value comparison',
            'Fig.1c. Alpha value comparion',
            ]
    for i, (fig, fig_text, ) \
    in enumerate(zip(figs, fig_texts, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        fig.text(0.1,
                0.05,
                fig_text,
                horizontalalignment='left',
                color=(0,0,0,1),
                family='Liberation Sans',
                fontweight=10,
                fontsize=10,
                transform=fig.transAxes,
                )


    # """
	# ~~~~Additional figures format~~~~
	# """
    # Format legend
    for ax in msd_figs:
        format_legend(ax,
                show_legend=True,
                legend_loc='upper left',
                legend_fontweight=5,
                legend_fontsize=5,
                )
    # # Rename legend
    # for ax in rename_msd_figs:
    #     rename_legend(ax,
    #             new_labels=['Boundary', 'Inside'],
    #             replace_ind=1,
    #             replace_type='prefix',
    #             legend_loc='lower right',
    #             legend_fontweight=7,
    #             legend_fontsize=7,
    #             )
    # Format scale
    figs = [
            axs[0],
            axs_s1[0], axs_s1[1], axs_s1[2],
            axs_s1[3], axs_s1[4], axs_s1[5],
            ]
    xscales = [
            [-0.05, 1.05],
            [0, 15000], [0,15000], [0, 15000],
            [0, 1], [0, 1], [0, 1],
            ]
    yscales = [
            [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],

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
    all_figures.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=600)
    plt.clf(); plt.close()
    # plt.show()
