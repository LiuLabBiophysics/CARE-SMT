import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from ..smt import get_d_values
from scipy.stats import norm, expon
import numpy as np
from ..qmath import t_test
import pandas as pd


def fig_quick_cilia_4(
    df_glb=pd.DataFrame([]),
    df_loc=pd.DataFrame([]),
    ):
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df_glb.empty and not df_loc.empty:
        # get df_particle, which drop duplicates of 'particle'
        df_glb_ptc = df_glb.drop_duplicates('particle')
        df_loc_ptc = df_loc.drop_duplicates('particle')

        df_glb_ptc= df_glb_ptc[ ~df_glb_ptc['raw_data'].isin(['cohort a sample 004c1-r-cilia2-physData.csv',
                                'cohort a sample 005c1-r-cilia2-physData.csv',
                                'cohort a sample 082c1-l-cilia3-physData.csv',
                                'cohort a sample 091c1-7-Cilia4-physData.csv']) ]

        df_loc_ptc= df_loc_ptc[ ~df_loc_ptc['raw_data'].isin(['cohort a sample 004c1-r-cilia2-physData.csv',
                                'cohort a sample 005c1-r-cilia2-physData.csv',
                                'cohort a sample 082c1-l-cilia3-physData.csv',
                                'cohort a sample 091c1-7-Cilia4-physData.csv']) ]

        # chosen_range = ( (df_loc_ptc['exp_label']=='cohort a')&(df_loc_ptc['D']>50000) )
        # df_loc_ptc = df_loc_ptc[ chosen_range ]
        # print(df_loc_ptc[['D', 'raw_data']])

        df_glb_ptc_a = df_glb_ptc[ df_glb_ptc['exp_label']=='cohort a' ]
        df_glb_ptc_b = df_glb_ptc[ df_glb_ptc['exp_label']=='cohort b' ]
        df_loc_ptc_a = df_loc_ptc[ df_loc_ptc['exp_label']=='cohort a' ]
        df_loc_ptc_b = df_loc_ptc[ df_loc_ptc['exp_label']=='cohort b' ]

        # add 'dist_to_base', bin_size=0.1
        df_glb['dist_to_base'] = round(df_glb['h_norm'], 1)

        df_glb_dropna = df_glb.dropna()
        df_glb_dropna.loc[df_glb_dropna['h_norm']>=0.8, 'location'] = 'tip'
        df_glb_dropna.loc[(df_glb_dropna['h_norm']>0.2)&(df_glb['h_norm']<0.8), 'location'] = 'middle'
        df_glb_dropna.loc[df_glb_dropna['h_norm']<=0.2, 'location'] = 'base'
        df_glb_dropna_a = df_glb_dropna[ df_glb_dropna['exp_label']=='cohort a' ]
        df_glb_dropna_b = df_glb_dropna[ df_glb_dropna['exp_label']=='cohort b' ]

        # df_glb_tip = df_glb_dropna[ df_glb_dropna['h_norm']>=0.8 ]
        # df_glb_base = df_glb_dropna[ df_glb_dropna['h_norm']<=0.2 ]
        # df_glb_middle = df_glb_dropna[ (df_glb_dropna['h_norm']>0.2)&(df_glb['h_norm']<0.8) ]
        # df_tip_a = df_glb_tip[ df_glb_tip['exp_label']=='cohort a' ]
        # df_tip_b = df_glb_tip[ df_glb_tip['exp_label']=='cohort b' ]
        # df_middle_a = df_glb_middle[ df_glb_middle['exp_label']=='cohort a' ]
        # df_middle_b = df_glb_middle[ df_glb_middle['exp_label']=='cohort b' ]
        # df_base_a = df_glb_base[ df_glb_base['exp_label']=='cohort a' ]
        # df_base_b = df_glb_base[ df_glb_base['exp_label']=='cohort b' ]

        df_tip_lifetime = df_glb[ df_glb['tip_lifetime']!=0 ]
        df_middle_lifetime = df_glb[ df_glb['middle_lifetime']!=0 ]
        df_base_lifetime = df_glb[ df_glb['base_lifetime']!=0 ]
        df_tip_lifetime_a = df_tip_lifetime[ df_tip_lifetime['exp_label']=='cohort a']
        df_tip_lifetime_b = df_tip_lifetime[ df_tip_lifetime['exp_label']=='cohort b']
        df_middle_lifetime_a = df_middle_lifetime[ df_middle_lifetime['exp_label']=='cohort a']
        df_middle_lifetime_b = df_middle_lifetime[ df_middle_lifetime['exp_label']=='cohort b']
        df_base_lifetime_a = df_base_lifetime[ df_base_lifetime['exp_label']=='cohort a']
        df_base_lifetime_b = df_base_lifetime[ df_base_lifetime['exp_label']=='cohort b']

        df_tip_lifetime['location'] = 'tip'
        df_tip_lifetime['lifetime'] = df_tip_lifetime['tip_lifetime']
        df_lt_tip = df_tip_lifetime[['exp_label', 'location', 'lifetime']]
        df_middle_lifetime['location'] = 'middle'
        df_middle_lifetime['lifetime'] = df_middle_lifetime['middle_lifetime']
        df_lt_middle = df_middle_lifetime[['exp_label', 'location', 'lifetime']]
        df_base_lifetime['location'] = 'base'
        df_base_lifetime['lifetime'] = df_base_lifetime['base_lifetime']
        df_lt_base = df_base_lifetime[['exp_label', 'location', 'lifetime']]
        df_lt = pd.concat([df_lt_tip, df_lt_middle, df_lt_base], ignore_index=True)
        print(df_lt)
        print(len(df_lt_tip), len(df_lt_middle), len(df_lt_base), len(df_lt))






    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(8.5, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.9, 1])

    fig1 = left_page.inset_axes([0, 0.78, 1, 0.22])
    fig2 = left_page.inset_axes([0, 0.56, 1, 0.22])
    fig3 = left_page.inset_axes([0, 0.34, 1, 0.22])

    fig1_1 = fig1.inset_axes([0.13, 0.675, 0.3, 0.3])
    fig1_2 = fig1.inset_axes([0.6, 0.675, 0.3, 0.3])
    fig1_3 = fig1.inset_axes([0.13, 0.175, 0.3, 0.3])
    # fig1_4 = fig1.inset_axes([0.6, 0.175, 0.3, 0.3])

    fig2_1 = fig2.inset_axes([0.13, 0.675, 0.3, 0.3])
    fig2_2 = fig2.inset_axes([0.6, 0.675, 0.3, 0.3])
    fig2_3 = fig2.inset_axes([0.13, 0.175, 0.3, 0.3])

    fig3_1 = fig3.inset_axes([0.13, 0.675, 0.3, 0.3])
    fig3_3 = fig3.inset_axes([0.6, 0.825, 0.3, 0.1])
    fig3_4 = fig3.inset_axes([0.6, 0.675, 0.3, 0.1], sharex=fig3_3)
    fig3_5 = fig3.inset_axes([0.13, 0.325, 0.3, 0.1])
    fig3_6 = fig3.inset_axes([0.13, 0.175, 0.3, 0.1], sharex=fig3_5)
    fig3_7 = fig3.inset_axes([0.6, 0.325, 0.3, 0.1])
    fig3_8 = fig3.inset_axes([0.6, 0.175, 0.3, 0.1], sharex=fig3_7)



    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, whole_page,
                fig1, fig2, fig3]:
        axis.set_xticks([]); axis.set_yticks([])

    # for axis in [fig5_1, fig5_3, fig6_1, fig6_3,
    #             fig7_3, fig7_5, fig7_7,
    #             fig3_3, fig3_5, fig3_7]:
    #     axis.set_xticks([])


    # """
	# ~~~~Plot D~~~~
	# """
    figs = [fig1_1]
    datas = [df_loc_ptc]
    xlabels = ['']
    ylabels = ['normalized D (a.u)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='exp_label',
                    y='D',
                    data=datas[i],
                    fliersize=2,
                    )
        sns.swarmplot(ax=figs[i],
                    x='exp_label',
                    y='D',
                    data=datas[i],
                    color=".25",
                    size=4,
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i],
                    cat_col='exp_label',
                    hist_col='D',
                    drop_duplicates=False,
                    text_pos=[0.5, 0.8],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=9,
                    horizontalalignment='center',
                    )

    # csv1_1 = df_loc_ptc[['exp_label', 'D']]
    # csv1_1.round(2).to_csv('/home/linhua/Desktop/Data1_1.csv',
                    # index=False)

    # """
	# ~~~~Plot alpha~~~~
	# """
    figs = [fig1_2]
    datas = [df_loc_ptc]
    xlabels = ['']
    ylabels = ['alpha']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='exp_label',
                    y='alpha',
                    data=datas[i],
                    fliersize=2,
                    )
        sns.swarmplot(ax=figs[i],
                    x='exp_label',
                    y='alpha',
                    data=datas[i],
                    color=".25",
                    size=4,
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i],
                    cat_col='exp_label',
                    hist_col='alpha',
                    drop_duplicates=False,
                    text_pos=[0.5, 0.8],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=9,
                    horizontalalignment='center',
                    )

    # csv1_2 = df_loc_ptc[['exp_label', 'alpha']]
    # csv1_2.round(2).to_csv('/home/linhua/Desktop/Data1_2.csv',
                    # index=False)


    # """
	# ~~~~Plot speed~~~~
	# """
    figs = [fig2_1]
    datas = [df_glb_dropna]
    xlabels = ['normalized distance to base']
    ylabels = ['speed (a.u)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.lineplot(ax=figs[i],
                    x='dist_to_base',
                    y='v_norm_abs',
                    hue='exp_label',
                    data=datas[i],
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )

    # csv2_1 = df_glb_dropna[['exp_label', 'dist_to_base', 'v_norm_abs']]
    # csv2_1 = csv2_1.rename(columns={'v_norm_abs':'speed'})
    # csv2_1.round(2).to_csv('/home/linhua/Desktop/Data2_1.csv',
    #                 index=False)



    # """
	# ~~~~Plot speed boxplot~~~~
	# """
    figs = [fig2_2, fig2_3]
    datas = [df_glb_dropna_a, df_glb_dropna_b]
    xlabels = ['cohort a', 'cohort b']
    ylabels = ['speed (a.u)', 'speed (a.u)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='location',
                    y='v_norm_abs',
                    data=datas[i],
                    order=['base', 'middle', 'tip'],
                    fliersize=2,
                    )
        sns.swarmplot(ax=figs[i],
                    x='location',
                    y='v_norm_abs',
                    data=datas[i],
                    order=['base', 'middle', 'tip'],
                    color=".25",
                    size=1,
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ datas[i]['location']!='tip' ],
                    cat_col='location',
                    hist_col='v_norm_abs',
                    drop_duplicates=False,
                    text_pos=[0.35, 0.9],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=6,
                    horizontalalignment='center',
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ datas[i]['location']!='base' ],
                    cat_col='location',
                    hist_col='v_norm_abs',
                    drop_duplicates=False,
                    text_pos=[0.65, 0.9],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=6,
                    horizontalalignment='center',
                    )
    # csv2_2 = df_glb_dropna_a[['exp_label', 'location', 'v_norm_abs']]
    # csv2_2 = csv2_2.rename(columns={'v_norm_abs':'speed'})
    # csv2_2.round(3).to_csv('/home/linhua/Desktop/Data2_2.csv',
    #                 index=False)
    #
    # csv2_3 = df_glb_dropna_b[['exp_label', 'location', 'v_norm_abs']]
    # csv2_3 = csv2_3.rename(columns={'v_norm_abs':'speed'})
    # csv2_3.round(3).to_csv('/home/linhua/Desktop/Data2_3.csv',
    #                 index=False)


    # """
	# ~~~~Plot lifetime boxplot~~~~
	# """
    figs = [fig3_1]
    datas = [df_lt]
    xlabels = ['']
    ylabels = ['lifetime (s)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='exp_label',
                    y='lifetime',
                    hue='location',
                    data=datas[i],
                    order=['cohort a', 'cohort b'],
                    hue_order=['base', 'middle', 'tip'],
                    fliersize=2,
                    whis=15,
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ (datas[i]['exp_label']!='cohort a') & (datas[i]['location']!='tip') ],
                    cat_col='location',
                    hist_col='lifetime',
                    drop_duplicates=False,
                    text_pos=[0.125, 0.5],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=5,
                    horizontalalignment='center',
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ (datas[i]['exp_label']!='cohort a') & (datas[i]['location']!='base') ],
                    cat_col='location',
                    hist_col='lifetime',
                    drop_duplicates=False,
                    text_pos=[0.37, 0.5],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=5,
                    horizontalalignment='center',
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ (datas[i]['exp_label']!='cohort b') & (datas[i]['location']!='tip') ],
                    cat_col='location',
                    hist_col='lifetime',
                    drop_duplicates=False,
                    text_pos=[0.625, 0.4],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=5,
                    horizontalalignment='center',
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i][ (datas[i]['exp_label']!='cohort b') & (datas[i]['location']!='base') ],
                    cat_col='location',
                    hist_col='lifetime',
                    drop_duplicates=False,
                    text_pos=[0.87, 0.4],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=5,
                    horizontalalignment='center',
                    )

    # csv3_1 = df_lt[['exp_label', 'location', 'lifetime']]
    # csv3_1.round(2).to_csv('/home/linhua/Desktop/Data3_1.csv',
    #                 index=False)


    # """
	# ~~~~Plot hist~~~~
	# """
    current_palette = sns.color_palette()
    c1 = current_palette[0]
    c2 = current_palette[1]
    figs = [fig3_3, fig3_4, fig3_5, fig3_6, fig3_7, fig3_8,
            ]
    colors = [c1, c2, c1, c2, c1, c2,
            ]
    datas = [
            df_tip_lifetime_a, df_tip_lifetime_b, df_middle_lifetime_a,
            df_middle_lifetime_b, df_base_lifetime_a, df_base_lifetime_b,
            ]
    cols = [
            'tip_lifetime', 'tip_lifetime', 'middle_lifetime',
            'middle_lifetime', 'base_lifetime', 'base_lifetime',
            ]
    bins = [
            10, 5, 20, 15, 5, 5,
            ]
    notes = ['cohort a', 'cohort b', 'cohort a', 'cohort b', 'cohort a', 'cohort b',]
    fits = [
            None, None, None, None, None, None,
            # expon, expon, expon, expon, expon, expon,
            ]
    xlabels = [
            '', 'lifetime at tip (s)', '', 'lifetime at middle (s)', '', 'lifetime at base (s)',
            ]
    ylabels = [
            'counts', 'counts', 'counts', 'counts', 'counts', 'counts']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.distplot(datas[i][cols[i]],
                    bins=bins[i],
                    kde=False,
                    color=colors[i],
                    hist_kws={'linewidth': 0.5, 'edgecolor': (0,0,0)},
                    fit=fits[i],
                    fit_kws={'linewidth': 0.5, 'color': (0,0,0)},
                    ax=figs[i],
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        figs[i].text(0.98,
                0.7,
                notes[i],
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 6,
                color = (0, 0, 0, 1),
                transform=figs[i].transAxes,
                weight = 'normal',
                )
        # figs[i].text(0.95,
        #         0.2,
        #         """
        #         Fitting rslt: \n(%.2f, %.2f)
        #         """ %(fits[i].fit(datas[i][cols[i]])[0], fits[i].fit(datas[i][cols[i]])[1]),
        #         horizontalalignment='right',
        #         verticalalignment='bottom',
        #         fontsize = 6,
        #         color = (0, 0, 0, 1),
        #         transform=figs[i].transAxes,
        #         weight = 'normal',
        #         )

    # csv3_2_a = df_tip_lifetime_a[['exp_label', 'tip_lifetime']]
    # csv3_2_a.round(2).to_csv('/home/linhua/Desktop/Data3_2_a.csv',
    #                 index=False)
    # csv3_2_b = df_tip_lifetime_b[['exp_label', 'tip_lifetime']]
    # csv3_2_b.round(2).to_csv('/home/linhua/Desktop/Data3_2_b.csv',
    #                 index=False)
    #
    # csv3_3_a = df_middle_lifetime_a[['exp_label', 'middle_lifetime']]
    # csv3_3_a.round(2).to_csv('/home/linhua/Desktop/Data3_3_a.csv',
    #                 index=False)
    # csv3_3_b = df_middle_lifetime_b[['exp_label', 'middle_lifetime']]
    # csv3_3_b.round(2).to_csv('/home/linhua/Desktop/Data3_3_b.csv',
    #                 index=False)
    #
    # csv3_4_a = df_base_lifetime_a[['exp_label', 'base_lifetime']]
    # csv3_4_a.round(2).to_csv('/home/linhua/Desktop/Data3_4_a.csv',
    #                 index=False)
    # csv3_4_b = df_base_lifetime_b[['exp_label', 'base_lifetime']]
    # csv3_4_b.round(2).to_csv('/home/linhua/Desktop/Data3_4_b.csv',
    #                 index=False)



    # """
	# ~~~~Plot restored mean MSD curve~~~~
	# """
    figs = [fig1_3, fig1_3]
    colors = [c1, c2]
    datas = [df_loc_ptc_a, df_loc_ptc_b]
    xlabels = ['', 'time (s)',]
    ylabels = ['', r'mean MSD (a.u)',]
    x = np.array(range(31))
    msds = []
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        D = datas[i]['D'].mean()
        alpha = datas[i]['alpha'].mean()
        mean_msd = 4*D*(x**alpha)
        msds.append(mean_msd)
        figs[i].plot(x, mean_msd, '-o',
                    markersize=2.5,
                    linewidth=1,
                    color=colors[i])

        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
    t_stats = t_test(msds[0], msds[1])
    t_test_str = 'P = %.1E' % (t_stats[1])
    figs[0].text(0.5,
            0.8,
            t_test_str,
            horizontalalignment='center',
            color=(0,0,0,1),
            family='Liberation Sans',
            fontweight=9,
            fontsize=9,
            transform=figs[0].transAxes)

    # csv1_3 = pd.DataFrame([], columns=['time', 'cohort a msd', 'cohort b msd'])
    # csv1_3['time'] = x
    # csv1_3['cohort a msd'] = msds[0]
    # csv1_3['cohort b msd'] = msds[1]
    # csv1_3.round(2).to_csv('/home/linhua/Desktop/Data1_3.csv',
    #                 index=False)




    # """
	# ~~~~Add figure text~~~~
	# """
    figs = [fig1, fig2, fig3]
    texts = [
            'Fig.1. Local oscillation mobility comparison',
            'Fig.2. Speed vs location',
            'Fig.3. Lifetime vs location',
            ]
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        figs[i].text(0.02,
                0.02,
                texts[i],
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize = 9,
                color = (0, 0, 0),
                transform=figs[i].transAxes,
                weight = 'normal',
                # fontname = 'Arial',
                )











    # """
	# ~~~~format figures~~~~
	# """
    # figs = [fig2_2, fig2_3]
    # y_min = [-0.02, -0.02]
    # y_max = [0.8, 0.8]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             yscale=[y_min[i], y_max[i]],
    #             )

    for figure in [fig1_1, fig1_2, fig1_3, #fig1_4,
                fig2_1, fig2_2, fig2_3,
                fig3_1, fig3_3, fig3_4, fig3_5, fig3_6, fig3_7, fig3_8,
                ]:
        format_spine(figure, spine_linewidth=0.5)
        format_tick(figure, tk_width=0.5)
        format_tklabel(figure, tklabel_fontsize=8)
        format_label(figure, label_fontsize=9)
    for figure in [fig2_1, fig3_1]:
        format_legend(figure,
                show_legend=True,
                legend_loc='upper center',
                legend_fontweight=6,
                legend_fontsize=6,
                )




    for figure in [fig1_1, fig1_2, fig1_3,
                fig2_1, fig2_2, fig2_3,
                fig3_1, fig3_3, fig3_4, fig3_5, fig3_6, fig3_7, fig3_8,
                ]:
        figure.set_xlabel('')
        figure.set_ylabel('')
        for txt in figure.texts:
            txt.set_visible(False)

    for figure in [fig1_1, fig1_2,
                fig2_2, fig2_3,
                fig3_1,
                ]:
        # figure.get_xaxis().set_ticks([])
        labels = [item.get_text() for item in figure.get_xticklabels()]
        empty_string_labels = ['']*len(labels)
        figure.set_xticklabels(empty_string_labels)
















    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.tiff', dpi=600)
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
