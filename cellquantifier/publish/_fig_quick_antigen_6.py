import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2
from ..phys import *

def fig_quick_antigen_6(
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
	# ~~~~Prepare df for the whole page~~~~
	# """
    print("\n")
    print("Preparing data")
    if not df.empty:
        # """
        # ~~~~Filters applied to df~~~~
        # """
        df = df[ df['traj_length']>=20 ]

        df['subparticle_traj_length'] = df['subparticle_traj_length'] * 0.5
        df['subparticle_travel_dist'] = df['subparticle_travel_dist'] * 0.163

        raw_datas = df['raw_data'].unique()
        for raw_data in raw_datas:
            curr_df = df[ df['raw_data']==raw_data ]
            DM_df = curr_df[ curr_df['subparticle_final_type']=='final_DM']

            DM_subtraj_num = DM_df['subparticle'].nunique()
            DM_traj_num = DM_df['particle'].nunique()
            traj_num = curr_df['particle'].nunique()

            frame_num = curr_df['frame'].nunique()
            tot_foci_num = len(curr_df)
            foci_num_mean = tot_foci_num / frame_num

            df.loc[df['raw_data']==raw_data, 'DM_subtraj_num'] = DM_subtraj_num
            df.loc[df['raw_data']==raw_data, 'DM_traj_num'] = DM_traj_num
            df.loc[df['raw_data']==raw_data, 'traj_num'] = traj_num
            df.loc[df['raw_data']==raw_data, 'foci_num_mean'] = foci_num_mean

            df.loc[df['raw_data']==raw_data, 'DM_traj_ratio1'] = DM_traj_num / foci_num_mean
            df.loc[df['raw_data']==raw_data, 'DM_traj_ratio2'] = DM_traj_num / traj_num
            df.loc[df['raw_data']==raw_data, 'DM_subtraj_ratio1'] = DM_subtraj_num / foci_num_mean
            df.loc[df['raw_data']==raw_data, 'DM_subtraj_ratio2'] = DM_subtraj_num / traj_num

        # dfr is df_rawdata
        dfr = df[['exp_label', 'raw_data', 'DM_subtraj_num', 'DM_traj_num',
            'foci_num_mean', 'traj_num',
            'DM_traj_ratio1', 'DM_traj_ratio2',
            'DM_subtraj_ratio1', 'DM_subtraj_ratio2']].drop_duplicates('raw_data')
        dfr = dfr.reset_index(drop=True)
        print(dfr.to_string())
        print( ((dfr[['exp_label', 'DM_subtraj_num', 'DM_traj_num',
            'foci_num_mean', 'traj_num',
            'DM_traj_ratio1', 'DM_traj_ratio2',
            'DM_subtraj_ratio1', 'DM_subtraj_ratio2']].groupby(['exp_label'])).mean()).to_string() )
        # prepare dfr for t-test
        dfr_OEWT = dfr[ dfr['exp_label']!='MalKN' ]
        dfr_WTKN = dfr[ dfr['exp_label']!='MalOE' ]

        # dfsp is df_subparticle
        dfsp_CM = df[ df['subparticle_final_type']=='final_CM' ].drop_duplicates('subparticle')
        dfsp_BM = df[ df['subparticle_final_type']=='final_BM' ].drop_duplicates('subparticle')
        dfsp_DM = df[ df['subparticle_final_type']=='final_DM' ].drop_duplicates('subparticle')
        # prepare dfsp for t-test
        dfsp_DM_OEWT = dfsp_DM[ dfsp_DM['exp_label']!='MalKN' ]
        dfsp_DM_WTKN = dfsp_DM[ dfsp_DM['exp_label']!='MalOE' ]


    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 3
    row_num = 3
    divide_index = [
        ]
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
	# ~~~~Plot boxplot~~~~
	# """
    figs = axs
    datas = [
            dfr, dfr, dfr,
            dfsp_DM, dfsp_DM, dfsp_DM,
            dfsp_DM, dfsp_DM,
            ]
    data_cols = [
            'DM_traj_num', 'foci_num_mean', 'DM_traj_ratio1',
            'subparticle_traj_length', 'subparticle_v', 'subparticle_travel_dist',
            'subparticle_alpha', 'subparticle_dir_pers',
            ]
    palettes = [
            p, p, p, p, p, p, p, p, p,
            ]
    orders = [
            ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
            ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
            ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
            ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
            ['MalOE', 'WT', 'MalKN'],
            ]
    xlabels = [
            '', '', '',
            '', '', '',
            '', '', '',
            ]
    ylabels = [
            'DM_traj_num', 'foci_num', r'$\frac{DM\_traj\_num}{foci\_num}$',
            'DM_lifetime (s)', 'DM_velocity (nm/s)', 'DM_travel_distance (um)',
            'DM_alpha', 'DM_directional_persistence',
            ]
    for i, (fig, data, data_col, palette, order, xlabel, ylabel,) \
    in enumerate(zip(figs, datas, data_cols, palettes, orders, xlabels, ylabels,)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        if fig not in [axs[0], axs[1], axs[2]]:
            filersize = 2
            swarmsize = 1.5
        else:
            filersize = 2
            swarmsize = 3

        sns.boxplot(ax=fig,
                    x='exp_label',
                    y=data_col,
                    data=data,
                    order=order,
                    palette=palette,
                    linewidth=1,
                    boxprops=dict(alpha=RGBA_alpha, linewidth=1, edgecolor=(0,0,0)),
                    saturation=1,
                    fliersize=filersize,
                    # whis=[0, 100],
                    whis=1.5,
                    )

        sns.swarmplot(ax=fig,
                    x='exp_label',
                    y=data_col,
                    data=data,
                    order=order,
                    color="0",
                    size=swarmsize,
                    )

        set_xylabel(fig,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    )

    # # """
	# # ~~~~Add t test~~~~
	# # """
    # figs = [
    #         axs[0], axs[0],
    #         axs[1], axs[1],
    #         axs[2], axs[2],
    #         axs[3], axs[3],
    #         ]
    # datas = [
    #         dfr_OEWT, dfr_WTKN,
    #         dfr_OEWT, dfr_WTKN,
    #         dfr_OEWT, dfr_WTKN,
    #         dfsp_DM_OEWT, dfsp_DM_WTKN,
    #         ]
    # data_cols = [
    #         'DM_traj_num', 'DM_traj_num',
    #         'foci_num_mean', 'foci_num_mean',
    #         'DM_traj_ratio1', 'DM_traj_ratio1',
    #         'subparticle_traj_length', 'subparticle_traj_length',
    #         ]
    # cat_cols = [
    #         'exp_label', 'exp_label',
    #         'exp_label', 'exp_label',
    #         'exp_label', 'exp_label',
    #         'exp_label', 'exp_label',
    #         ]
    # text_poss = [
    #         (0.45, 0.8), (0.8, 0.75),
    #         (0.45, 0.8), (0.8, 0.75),
    #         (0.45, 0.8), (0.8, 0.75),
    #         (0.45, 0.8), (0.8, 0.75),
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
    #                 format='general',
    #                 )



    # # Format scale
    # figs = [ axs[6], axs[7], axs[8] ]
    # xscales = [
    #         [None, None], [None, None], [None, None],
    #         ]
    # yscales = [
    #         [18, 110], [8, 30], [8, 50],
    #         ]
    # for i, (fig, xscale, yscale, ) \
    # in enumerate(zip(figs, xscales, yscales,)):
    #     format_scale(fig,
    #             xscale=xscale,
    #             yscale=yscale,
    #             )

    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
	# """
    all_figures.savefig('/home/linhua/Desktop/antigen-comparison.pdf', dpi=300)
    plt.clf(); plt.close()
