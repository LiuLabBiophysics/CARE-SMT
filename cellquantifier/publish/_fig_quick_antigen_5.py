import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2
from ..phys import *

def fig_quick_antigen_5(
    df=pd.DataFrame([]),
    ):

    # """
	# ~~~~Initialize the colors~~~~
	# """
    print("\n")
    print("Preparing colors")
    # palette = sns.color_palette('muted')
    # c1 = palette[0]
    # c2 = palette[1]
    # c3 = palette[2]
    c1 = (1, 0, 0)
    c2 = (0, 0, 0)
    c3 = (0, 0, 1)
    c4 = (0.5, 0.5, 0.5)
    RGBA_alpha = 0.7
    c1_alp = (c1[0], c1[1], c1[2], RGBA_alpha)
    c2_alp = (c2[0], c2[1], c2[2], RGBA_alpha)
    c3_alp = (c3[0], c3[1], c3[2], RGBA_alpha)

    p = [c1, c2, c3]

    p1 = [c2, c4]

    pa = [c3, c2]
    pb = [c1, c2]


    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 4
    row_num = 6
    divide_index = [
        ]
    hidden_index = []
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 3
    index_s1 = [
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
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
        df['subparticle'] = df['raw_data'] + df['subparticle']
        dfp = df.drop_duplicates('subparticle')

        dfp_OE = dfp[ dfp['exp_label']=='MalOE' ]
        dfp_WT = dfp[ dfp['exp_label']=='WT' ]
        dfp_KN = dfp[ dfp['exp_label']=='MalKN' ]

        dfp_OE_BM = dfp_OE[ dfp_OE['subparticle_final_type']=='final_BM' ]
        dfp_OE_CM = dfp_OE[ dfp_OE['subparticle_final_type']=='final_CM' ]
        dfp_OE_DM = dfp_OE[ dfp_OE['subparticle_final_type']=='final_DM' ]
        dfp_WT_BM = dfp_WT[ dfp_WT['subparticle_final_type']=='final_BM' ]
        dfp_WT_CM = dfp_WT[ dfp_WT['subparticle_final_type']=='final_CM' ]
        dfp_WT_DM = dfp_WT[ dfp_WT['subparticle_final_type']=='final_DM' ]
        dfp_KN_BM = dfp_KN[ dfp_KN['subparticle_final_type']=='final_BM' ]
        dfp_KN_CM = dfp_KN[ dfp_KN['subparticle_final_type']=='final_CM' ]
        dfp_KN_DM = dfp_KN[ dfp_KN['subparticle_final_type']=='final_DM' ]


        df = df.drop('particle', axis=1)
        df = df.rename(columns={'subparticle':'particle',})
        df_BM = df[ df['subparticle_final_type']=='final_BM' ]
        df_CM = df[ df['subparticle_final_type']=='final_CM' ]
        df_DM = df[ df['subparticle_final_type']=='final_DM' ]

    # """
	# ~~~~Plot msd~~~~
	# """
    msd_figs = [
        axs[0], axs[1], axs[2], axs[3],
        ]
    datas = [
        df, df_CM, df_BM, df_DM,
        ]
    palettes = [
        p, p, p, p,
        ]
    cat_cols = [
        'exp_label', 'exp_label', 'exp_label', 'exp_label',
        ]
    orders = [
        ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
        ]

    for i, (fig, data, palette, cat_col, order, ) \
    in enumerate(zip(msd_figs, datas, palettes, cat_cols, orders, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(msd_figs)))

        add_mean_msd2(fig, data,
                    pixel_size=0.163,
                    frame_rate=2,
                    divide_num=5,

                    cat_col=cat_col,
                    cat_order=order,
                    color_order=palette,
                    RGBA_alpha=RGBA_alpha,

                    fitting_linewidth=1,
                    elinewidth=None,
                    markersize=None,
                    capsize=2,
                    )
        set_xylabel(fig,
                    xlabel='Time (s)',
                    ylabel=r'MSD (nm$^2$)',
                    )


    # # """
	# # ~~~~Plot particle number boxplot~~~~
	# # """
    # figs = [
    #         axs[18], axs[19], axs[20],
    #         axs[22], axs[23],
    #         ]
    # datas = [
    #         dmr, dmrb, dmra,
    #         dmrb_OEWT, dmra_KNWT,
    #         ]
    # palettes = [
    #         p, p, p,
    #         pb, pa,
    #         ]
    # orders = [
    #         ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
    #         ['MalOE', 'WT'], ['MalKN', 'WT'],
    #         ]
    # xlabels = [
    #         '', '', '',
    #         '', '',
    #         ]
    # ylabels = [
    #         'Antigen num per cell', 'Antigen num per cell', 'Antigen num cell',
    #         'Antigen num per cell', 'Antigen num cell',
    #         ]
    # for i, (fig, data, palette, order, xlabel, ylabel,) \
    # in enumerate(zip(figs, datas, palettes, orders, xlabels, ylabels,)):
    #     print("\n")
    #     print("Plotting (%d/%d)" % (i+1, len(figs)))
    #     sns.boxplot(ax=fig,
    #                 x='exp_label',
    #                 y='particle_num',
    #                 data=data,
    #                 order=order,
    #                 palette=palette,
    #                 boxprops=dict(alpha=RGBA_alpha),
    #                 saturation=1,
    #                 fliersize=2,
    #                 whis=5,
    #                 )
    #     sns.swarmplot(ax=fig,
    #                 x='exp_label',
    #                 y='particle_num',
    #                 data=data,
    #                 order=order,
    #                 color="0",
    #                 size=4,
    #                 )
    #     set_xylabel(fig,
    #                 xlabel=xlabel,
    #                 ylabel=ylabel,
    #                 )

    # """
	# ~~~~Plot hist~~~~
	# """
    figs = axs_s1
    datas = [
            dfp_OE, dfp_WT, dfp_KN,
            dfp_OE_CM, dfp_WT_CM, dfp_KN_CM,
            dfp_OE_BM, dfp_WT_BM, dfp_KN_BM,
            dfp_OE_DM, dfp_WT_DM, dfp_KN_DM,

            dfp_OE, dfp_WT, dfp_KN,
            dfp_OE_CM, dfp_WT_CM, dfp_KN_CM,
            dfp_OE_BM, dfp_WT_BM, dfp_KN_BM,
            dfp_OE_DM, dfp_WT_DM, dfp_KN_DM,

            dfp_OE, dfp_WT, dfp_KN,
            dfp_OE_CM, dfp_WT_CM, dfp_KN_CM,
            dfp_OE_BM, dfp_WT_BM, dfp_KN_BM,
            dfp_OE_DM, dfp_WT_DM, dfp_KN_DM,

            dfp_OE, dfp_WT, dfp_KN,
            dfp_OE_CM, dfp_WT_CM, dfp_KN_CM,
            dfp_OE_BM, dfp_WT_BM, dfp_KN_BM,
            dfp_OE_DM, dfp_WT_DM, dfp_KN_DM,

            dfp_OE, dfp_WT, dfp_KN,
            dfp_OE_CM, dfp_WT_CM, dfp_KN_CM,
            dfp_OE_BM, dfp_WT_BM, dfp_KN_BM,
            dfp_OE_DM, dfp_WT_DM, dfp_KN_DM,
            ]
    bins = [
            None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,

            None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,

            None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,

            None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,

            None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,
            ]
    data_cols = [
            'subparticle_D', 'subparticle_D', 'subparticle_D',
            'subparticle_D', 'subparticle_D', 'subparticle_D',
            'subparticle_D', 'subparticle_D', 'subparticle_D',
            'subparticle_D', 'subparticle_D', 'subparticle_D',

            'subparticle_alpha', 'subparticle_alpha', 'subparticle_alpha',
            'subparticle_alpha', 'subparticle_alpha', 'subparticle_alpha',
            'subparticle_alpha', 'subparticle_alpha', 'subparticle_alpha',
            'subparticle_alpha', 'subparticle_alpha', 'subparticle_alpha',

            'subparticle_dir_pers', 'subparticle_dir_pers', 'subparticle_dir_pers',
            'subparticle_dir_pers', 'subparticle_dir_pers', 'subparticle_dir_pers',
            'subparticle_dir_pers', 'subparticle_dir_pers', 'subparticle_dir_pers',
            'subparticle_dir_pers', 'subparticle_dir_pers', 'subparticle_dir_pers',

            'subparticle_traj_length', 'subparticle_traj_length', 'subparticle_traj_length',
            'subparticle_traj_length', 'subparticle_traj_length', 'subparticle_traj_length',
            'subparticle_traj_length', 'subparticle_traj_length', 'subparticle_traj_length',
            'subparticle_traj_length', 'subparticle_traj_length', 'subparticle_traj_length',

            'subparticle_travel_dist', 'subparticle_travel_dist', 'subparticle_travel_dist',
            'subparticle_travel_dist', 'subparticle_travel_dist', 'subparticle_travel_dist',
            'subparticle_travel_dist', 'subparticle_travel_dist', 'subparticle_travel_dist',
            'subparticle_travel_dist', 'subparticle_travel_dist', 'subparticle_travel_dist',
            ]
    colors = [
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,

            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,

            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,

            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,

            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            c1, c2, c3,
            ]
    for i, (fig, data, bin, data_col, color, ) \
    in enumerate(zip(figs, datas, bins, data_cols, colors, )):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        sns.distplot(data[data_col],
                    bins=bin,
                    kde=False,
                    color=color,
                    ax=fig,
                    hist_kws={"alpha": RGBA_alpha,
                    'linewidth': 0.5, 'edgecolor': (0,0,0)},
                    )


    # # """
	# # ~~~~Add t test~~~~
	# # """
    # figs = [
    #         axs_s1[0], axs_s1[2], axs_s1[3], axs_s1[5],
    #         axs_s1[6], axs_s1[8], axs_s1[9], axs_s1[11],
    #         axs_s1[12], axs_s1[14], axs_s1[15], axs_s1[17],
    #
    #         axs[22], axs[23],
    #
    #         axs_s2[0], axs_s2[2],
    #         axs_s2[4], axs_s2[6],
    #         axs_s2[8], axs_s2[10],
    #
    #         axs_s2[12], axs_s2[14], axs_s2[16], axs_s2[18],
    #         axs_s2[20], axs_s2[22], axs_s2[24], axs_s2[26],
    #         ]
    # datas = [
    #         df1p_OEWT, df1p_KNWT, df1p_OEWT, df1p_KNWT,
    #         df2p_OEWT, df2p_KNWT, df2p_OEWT, df2p_KNWT,
    #         df3p_OEWT, df3p_KNWT, df3p_OEWT, df3p_KNWT,
    #
    #         dmrb_OEWT, dmra_KNWT,
    #
    #         dmp_OE, dmp_OE,
    #         dmp_WT, dmp_WT,
    #         dmp_KN, dmp_KN,
    #
    #         dmpb_OEWT, dmpb_OEWT, dmpa_OEWT, dmpa_OEWT,
    #         dmpa_KNWT, dmpa_KNWT, dmpb_KNWT, dmpb_KNWT,
    #         ]
    # data_cols = [
    #         'D', 'D', 'alpha', 'alpha',
    #         'D', 'D', 'alpha', 'alpha',
    #         'D', 'D', 'alpha', 'alpha',
    #
    #         'particle_num', 'particle_num',
    #
    #         'D', 'alpha',
    #         'D', 'alpha',
    #         'D', 'alpha',
    #
    #         'D', 'alpha', 'D', 'alpha',
    #         'D', 'alpha', 'D', 'alpha',
    #         ]
    # cat_cols = [
    #         'exp_label', 'exp_label', 'exp_label', 'exp_label',
    #         'exp_label', 'exp_label', 'exp_label', 'exp_label',
    #         'exp_label', 'exp_label', 'exp_label', 'exp_label',
    #
    #         'exp_label', 'exp_label',
    #
    #         'particle_type', 'particle_type',
    #         'particle_type', 'particle_type',
    #         'particle_type', 'particle_type',
    #
    #         'exp_label', 'exp_label', 'exp_label', 'exp_label',
    #         'exp_label', 'exp_label', 'exp_label', 'exp_label',
    #         ]
    # text_poss = [
    #         (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),
    #         (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),
    #         (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),
    #
    #         (0.98, 0.88), (0.98, 0.88),
    #
    #         (0.98, 0.78), (0.98, 0.78),
    #         (0.98, 0.78), (0.98, 0.78),
    #         (0.98, 0.78), (0.98, 0.78),
    #
    #         (0.98, 0.78), (0.98, 0.78), (0.98, 0.78), (0.98, 0.78),
    #         (0.98, 0.78), (0.98, 0.78), (0.98, 0.78), (0.98, 0.78),
    #         ]
    # for i, (fig, data, data_col, cat_col, text_pos, ) \
    # in enumerate(zip(figs, datas, data_cols, cat_cols, text_poss, )):
    #     print("\n")
    #     print("Plotting (%d/%d)" % (i+1, len(figs)))
    #
    #     add_t_test(fig,
    #                 blobs_df=data,
    #                 cat_col=cat_col,
    #                 hist_col=data_col,
    #                 drop_duplicates=False,
    #                 text_pos=text_pos,
    #                 color=(0,0,0,1),
    #                 fontname='Liberation Sans',
    #                 fontweight=9,
    #                 fontsize=9,
    #                 horizontalalignment='right',
    #                 mode='general',
    #                 )
    #
    # # """
	# # ~~~~Add figure text~~~~
	# # """
    # figs = [
    #         grids[0],
    #         ]
    # fig_texts = [
    #         'D',
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
    # Format legend
    for ax in msd_figs:
        format_legend(ax,
                show_legend=True,
                legend_loc='upper left',
                legend_fontweight=7,
                legend_fontsize=7,
                )
    # Format scale
    figs = axs_s1 + axs_s2
    xscales = [
            [0, 5000], [0, 5000], [0, 5000],
            [0, 4000], [0, 4000], [0, 4000],
            [0, 5000], [0, 5000], [0, 5000],
            [0, 40000], [0, 40000], [0, 40000],

            [0, 2], [0, 2], [0, 2],
            [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],
            [0.8, 1.2], [0.8, 1.2], [0.8, 1.2],
            [1.3, 1.9], [1.3, 1.9], [1.3, 1.9],

            [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1],
            [0, 1], [0, 1], [0, 1],

            [20, 100], [20, 100], [20, 100],
            [20, 100], [20, 100], [20, 100],
            [20, 100], [20, 100], [20, 100],
            [20, 100], [20, 100], [20, 100],

            [0, 30], [0, 30], [0, 30],
            [0, 3], [0, 3], [0, 3],
            [0, 5], [0, 5], [0, 5],
            [10, 30], [10, 30], [10, 30],
            ]
    yscales = [
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None],
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
    all_figures.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=300)
    plt.clf(); plt.close()
