import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2
from ..phys import *

def fig_quick_antigen_4(
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
    col_num = 3
    row_num = 12
    divide_index = [
        6, 7, 8,
        15, 16, 17,
        21, 22, 23,
        27, 28, 29,
        ]
    hidden_index = [21]
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 3
    index_s1 = [
        1, 2, 4, 5, 7, 8,
        ]
    # Sub_axs_2 settings
    col_num_s2 = 1
    row_num_s2 = 2
    index_s2 = [
        10, 11, 13, 14, 16, 17,
        25, 26, 28, 29, 31, 32, 34, 35,
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
            df = df[ df['traj_length']>=20 ]
            df1 = df.copy()
            df1p = df1.drop_duplicates('particle')
            df1p_OE = df1p[ df1p['exp_label']=='MalOE' ]
            df1p_WT = df1p[ df1p['exp_label']=='WT' ]
            df1p_KN = df1p[ df1p['exp_label']=='MalKN' ]
            df1p_OEWT = df1p[ df1p['exp_label']!='MalKN' ]
            df1p_KNWT = df1p[ df1p['exp_label']!='MalOE' ]

        # travel_dist filter
        if 'travel_dist' in df:
            travel_dist_min = 0
            travel_dist_max = 7

            df2 = df[ (df['travel_dist']>=travel_dist_min) & \
            					(df['travel_dist']<=travel_dist_max) ]
            df2p = df2.drop_duplicates('particle')
            df2p_OE = df2p[ df2p['exp_label']=='MalOE' ]
            df2p_WT = df2p[ df2p['exp_label']=='WT' ]
            df2p_KN = df2p[ df2p['exp_label']=='MalKN' ]
            df2p_OEWT = df2p[ df2p['exp_label']!='MalKN' ]
            df2p_KNWT = df2p[ df2p['exp_label']!='MalOE' ]
            df3 = df[ df['travel_dist']>travel_dist_max ]
            df3p = df3.drop_duplicates('particle')
            df3p_OE = df3p[ df3p['exp_label']=='MalOE' ]
            df3p_WT = df3p[ df3p['exp_label']=='WT' ]
            df3p_KN = df3p[ df3p['exp_label']=='MalKN' ]
            df3p_OEWT = df3p[ df3p['exp_label']!='MalKN' ]
            df3p_KNWT = df3p[ df3p['exp_label']!='MalOE' ]

            df = df[ (df['travel_dist']>=travel_dist_min) & \
            					(df['travel_dist']<=travel_dist_max) ]

        # add particle type filter
        if 'particle_type' in df:
        	df = df[ df['particle_type']!='--none--']


        # """
        # ~~~~Divide df into sub_dfs~~~~
        # """
        df['date'] = df['raw_data'].astype(str).str[0:6]

        # dfs for msd mean msd curve
        dm = df[ df['date'].isin(['200205']) ] #df_mal
        dm = add_particle_num(dm)
        dm_OE = dm[ dm['exp_label']=='MalOE' ]
        dm_WT = dm[ dm['exp_label']=='WT' ]
        dm_KN = dm[ dm['exp_label']=='MalKN' ]
        dma = dm[ dm['particle_type']=='A' ] #df_mal_A
        dma = add_particle_num(dma)
        dmb = dm[ dm['particle_type']=='B' ] #df_mal_B
        dmb = add_particle_num(dmb)

        # dfs for particle_num comparion between exp_label
        dmr = dm.drop_duplicates('raw_data')
        dmrb = dmb.drop_duplicates('raw_data')
        dmra = dma.drop_duplicates('raw_data')
        dmrb_OEWT = dmrb[ dmrb['exp_label']!='MalKN']
        dmra_KNWT = dmra[ dmra['exp_label']!='MalOE']

        # dfs D and alpha
        dmp = dm.drop_duplicates('particle')
        dmp_OE = dmp[ dmp['exp_label']=='MalOE' ]
        dmp_WT = dmp[ dmp['exp_label']=='WT' ]
        dmp_KN = dmp[ dmp['exp_label']=='MalKN' ]

        dmpa = dma.drop_duplicates('particle')
        dmpa_OE = dmpa[ dmpa['exp_label']=='MalOE' ]
        dmpa_WT = dmpa[ dmpa['exp_label']=='WT' ]
        dmpa_KN = dmpa[ dmpa['exp_label']=='MalKN' ]
        dmpa_OEWT = dmpa[ dmpa['exp_label']!='MalKN' ]
        dmpa_KNWT = dmpa[ dmpa['exp_label']!='MalOE' ]

        dmpb = dmb.drop_duplicates('particle')
        dmpb_OE = dmpb[ dmpb['exp_label']=='MalOE' ]
        dmpb_WT = dmpb[ dmpb['exp_label']=='WT' ]
        dmpb_KN = dmpb[ dmpb['exp_label']=='MalKN' ]
        dmpb_OEWT = dmpb[ dmpb['exp_label']!='MalKN' ]
        dmpb_KNWT = dmpb[ dmpb['exp_label']!='MalOE' ]


    # """
	# ~~~~Plot msd~~~~
	# """
    rename_msd_figs = [
        axs[9], axs[12], axs[15],
        ]
    msd_figs = [
        axs[0], axs[3], axs[6],
        axs[9], axs[12], axs[15],
        axs[24], axs[27], axs[30], axs[33],
        ]
    datas = [
        df1, df2, df3,
        dm_OE, dm_WT, dm_KN,
        dmb, dma, dma, dmb,
        ]
    palettes = [
        p, p, p,
        p1, p1, p1,
        pb, pb, pa, pa,
        ]
    cat_cols = [
        'exp_label', 'exp_label', 'exp_label',
        'particle_type', 'particle_type', 'particle_type',
        'exp_label', 'exp_label', 'exp_label', 'exp_label',
        ]
    orders = [
        ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
        ['A', 'B'], ['A', 'B'], ['A', 'B'],
        ['MalOE', 'WT'], ['MalOE', 'WT'], ['MalKN', 'WT'], ['MalKN', 'WT'],
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

                    fitting_linewidth=1.5,
                    )
        set_xylabel(fig,
                    xlabel='Time (s)',
                    ylabel=r'MSD (nm$^2$)',
                    )


    # """
	# ~~~~Plot particle number boxplot~~~~
	# """
    figs = [
            axs[18], axs[19], axs[20],
            axs[22], axs[23],
            ]
    datas = [
            dmr, dmrb, dmra,
            dmrb_OEWT, dmra_KNWT,
            ]
    palettes = [
            p, p, p,
            pb, pa,
            ]
    orders = [
            ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'], ['MalOE', 'WT', 'MalKN'],
            ['MalOE', 'WT'], ['MalKN', 'WT'],
            ]
    xlabels = [
            '', '', '',
            '', '',
            ]
    ylabels = [
            'Antigen num per cell', 'Antigen num per cell', 'Antigen num cell',
            'Antigen num per cell', 'Antigen num cell',
            ]
    for i, (fig, data, palette, order, xlabel, ylabel,) \
    in enumerate(zip(figs, datas, palettes, orders, xlabels, ylabels,)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=fig,
                    x='exp_label',
                    y='particle_num',
                    data=data,
                    order=order,
                    palette=palette,
                    boxprops=dict(alpha=RGBA_alpha),
                    saturation=1,
                    fliersize=2,
                    whis=5,
                    )
        sns.swarmplot(ax=fig,
                    x='exp_label',
                    y='particle_num',
                    data=data,
                    order=order,
                    color="0",
                    size=4,
                    )
        set_xylabel(fig,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    )

    # """
	# ~~~~Plot hist~~~~
	# """
    figs = axs_s1 + axs_s2
    datas = [
            df1p_OE, df1p_WT, df1p_KN, df1p_OE, df1p_WT, df1p_KN,
            df2p_OE, df2p_WT, df2p_KN, df2p_OE, df2p_WT, df2p_KN,
            df3p_OE, df3p_WT, df3p_KN, df3p_OE, df3p_WT, df3p_KN,

            dmpa_OE, dmpb_OE, dmpa_OE, dmpb_OE,
            dmpa_WT, dmpb_WT, dmpa_WT, dmpb_WT,
            dmpa_KN, dmpb_KN, dmpa_KN, dmpb_KN,

            dmpb_OE, dmpb_WT, dmpb_OE, dmpb_WT,
            dmpa_OE, dmpa_WT, dmpa_OE, dmpa_WT,
            dmpa_KN, dmpa_WT, dmpa_KN, dmpa_WT,
            dmpb_KN, dmpb_WT, dmpb_KN, dmpb_WT,
            ]
    bins = [
            150, None, 150, None, None, None,
            None, None, None, None, None, None,
            None, None, None, None, None, None,

            None, None, None, None,
            None, None, None, None,
            None, None, None, None,

            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            ]
    data_cols = [
            'D', 'D', 'D', 'alpha', 'alpha', 'alpha',
            'D', 'D', 'D', 'alpha', 'alpha', 'alpha',
            'D', 'D', 'D', 'alpha', 'alpha', 'alpha',

            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',

            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',
            ]
    colors = [
            c1, c2, c3, c1, c2, c3,
            c1, c2, c3, c1, c2, c3,
            c1, c2, c3, c1, c2, c3,

            c2, c4, c2, c4,
            c2, c4, c2, c4,
            c2, c4, c2, c4,

            c1, c2, c1, c2,
            c1, c2, c1, c2,
            c3, c2, c3, c2,
            c3, c2, c3, c2,
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


    # """
	# ~~~~Add t test~~~~
	# """
    figs = [
            axs_s1[0], axs_s1[2], axs_s1[3], axs_s1[5],
            axs_s1[6], axs_s1[8], axs_s1[9], axs_s1[11],
            axs_s1[12], axs_s1[14], axs_s1[15], axs_s1[17],

            axs[22], axs[23],

            axs_s2[0], axs_s2[2],
            axs_s2[4], axs_s2[6],
            axs_s2[8], axs_s2[10],

            axs_s2[12], axs_s2[14], axs_s2[16], axs_s2[18],
            axs_s2[20], axs_s2[22], axs_s2[24], axs_s2[26],
            ]
    datas = [
            df1p_OEWT, df1p_KNWT, df1p_OEWT, df1p_KNWT,
            df2p_OEWT, df2p_KNWT, df2p_OEWT, df2p_KNWT,
            df3p_OEWT, df3p_KNWT, df3p_OEWT, df3p_KNWT,

            dmrb_OEWT, dmra_KNWT,

            dmp_OE, dmp_OE,
            dmp_WT, dmp_WT,
            dmp_KN, dmp_KN,

            dmpb_OEWT, dmpb_OEWT, dmpa_OEWT, dmpa_OEWT,
            dmpa_KNWT, dmpa_KNWT, dmpb_KNWT, dmpb_KNWT,
            ]
    data_cols = [
            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',
            'D', 'D', 'alpha', 'alpha',

            'particle_num', 'particle_num',

            'D', 'alpha',
            'D', 'alpha',
            'D', 'alpha',

            'D', 'alpha', 'D', 'alpha',
            'D', 'alpha', 'D', 'alpha',
            ]
    cat_cols = [
            'exp_label', 'exp_label', 'exp_label', 'exp_label',
            'exp_label', 'exp_label', 'exp_label', 'exp_label',
            'exp_label', 'exp_label', 'exp_label', 'exp_label',

            'exp_label', 'exp_label',

            'particle_type', 'particle_type',
            'particle_type', 'particle_type',
            'particle_type', 'particle_type',

            'exp_label', 'exp_label', 'exp_label', 'exp_label',
            'exp_label', 'exp_label', 'exp_label', 'exp_label',
            ]
    text_poss = [
            (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),
            (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),
            (0.98, 0.68), (0.98, 0.68), (0.98, 0.68), (0.98, 0.68),

            (0.98, 0.88), (0.98, 0.88),

            (0.98, 0.78), (0.98, 0.78),
            (0.98, 0.78), (0.98, 0.78),
            (0.98, 0.78), (0.98, 0.78),

            (0.98, 0.78), (0.98, 0.78), (0.98, 0.78), (0.98, 0.78),
            (0.98, 0.78), (0.98, 0.78), (0.98, 0.78), (0.98, 0.78),
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
                    mode='general',
                    )

    # """
	# ~~~~Add figure text~~~~
	# """
    figs = [
            grids[0], grids[3], grids[6],
            grids[9], grids[12], grids[15],
            grids[18], grids[19], grids[20], grids[22], grids[23],
            grids[24], grids[25], grids[26], grids[27], grids[28], grids[29],
            grids[30], grids[31], grids[32], grids[33], grids[34], grids[35],
            ]
    fig_texts = [
            'Fig.1a. All particle comparison',
            'Fig.1b. Travel_dist <= 7px particle comparison',
            'Fig.1c. Travel_dist > 7px particle comparison',

            'Fig.2a. OE local movements comparion\nBoundary(A) vs Inside(B)',
            'Fig.2b. WT local movements comparion\nBoundary(A) vs Inside(B)',
            'Fig.2c. KN local movements comparion\nBoundary(A) vs Inside(B)',

            'Fig.3a. Antigen num (Boundary + Inside)',
            'Fig.3b. Antigen num (Inside)',
            'Fig.3c. Antigen num (Boundary)',
            'Fig.3d. Antigen num (Inside + OE&WT)',
            'Fig.3e. Antigen num (Boundary + KN&WT)',

            'Fig.4a. MSD (Inside + OE&WT)',
            'Fig.4b. D (Inside + OE&WT)',
            'Fig.4c. alpha (Inside + OE&WT)',
            'Fig.4d. MSD (Boundary + OE&WT)',
            'Fig.4e. D (Boundary + OE&WT)',
            'Fig.4f. alpha (Boundary + OE&WT)',

            'Fig.5a. MSD (Boundary + KN&WT)',
            'Fig.5b. D (Boundary + KN&WT)',
            'Fig.5c. alpha (Boundary + KN&WT)',
            'Fig.5d. MSD (Inside + KN&WT)',
            'Fig.5e. D (Inside + KN&WT)',
            'Fig.5f. alpha (Inside + KN&WT)',
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
                legend_loc='lower right',
                legend_fontweight=7,
                legend_fontsize=7,
                )
    # Rename legend
    for ax in rename_msd_figs:
        rename_legend(ax,
                new_labels=['Boundary', 'Inside'],
                replace_ind=1,
                replace_type='prefix',
                legend_loc='lower right',
                legend_fontweight=7,
                legend_fontsize=7,
                )
    # Format scale
    figs = axs_s1 + axs_s2
    xscales = [
            [0, 25000], [0, 25000], [0, 25000], [-0.5, 2], [-0.5, 2], [-0.5, 2],
            [0, 25000], [0, 25000], [0, 25000], [-0.5, 2], [-0.5, 2], [-0.5, 2],
            [0, 25000], [0, 25000], [0, 25000], [-0.5, 2], [-0.5, 2], [-0.5, 2],

            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],

            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            [0, 15000], [0, 15000], [-0.5, 2], [-0.5, 2],
            ]
    yscales = [
            [None, None], [None, None], [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None],

            [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None],
            [None, None], [None, None], [None, None], [None, None],
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
