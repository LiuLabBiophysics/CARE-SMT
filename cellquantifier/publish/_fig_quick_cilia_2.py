import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from ..smt import get_d_values


def fig_quick_cilia_2(
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

        # add 'dist_to_base', bin_size=0.1
        df_glb['dist_to_base'] = round(df_glb['h_norm'], 1)

        df_glb_dropna = df_glb.dropna()
        df_glb_tip = df_glb_dropna[ df_glb_dropna['h_norm']>=0.8 ]
        df_glb_base = df_glb_dropna[ df_glb_dropna['h_norm']<=0.2 ]
        df_glb_middle = df_glb_dropna[ (df_glb_dropna['h_norm']>0.2)&(df_glb['h_norm']<0.8) ]

        df_tip_lifetime = df_glb[ df_glb['tip_lifetime']!=0 ]
        df_middle_lifetime = df_glb[ df_glb['middle_lifetime']!=0 ]
        df_base_lifetime = df_glb[ df_glb['base_lifetime']!=0 ]





    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(17, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.89, 1, 0.11])
    fig2 = left_page.inset_axes([0, 0.78, 1, 0.11])
    fig3 = left_page.inset_axes([0, 0.56, 1, 0.22])
    fig4 = left_page.inset_axes([0, 0.34, 1, 0.22])

    fig1_1 = fig1.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig1_2 = fig1.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig2_1 = fig2.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig2_2 = fig2.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig3_1 = fig3.inset_axes([0.13, 0.675, 0.3, 0.3])
    fig3_2 = fig3.inset_axes([0.6, 0.675, 0.3, 0.3])
    fig3_3 = fig3.inset_axes([0.13, 0.175, 0.3, 0.3])
    fig3_4 = fig3.inset_axes([0.6, 0.175, 0.3, 0.3])

    fig4_2 = fig4.inset_axes([0.6, 0.675, 0.3, 0.3])
    fig4_3 = fig4.inset_axes([0.13, 0.175, 0.3, 0.3])
    fig4_4 = fig4.inset_axes([0.6, 0.175, 0.3, 0.3])



    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1, fig2, fig3, fig4]:
        axis.set_xticks([]); axis.set_yticks([])


    # """
	# ~~~~Plot D~~~~
	# """
    figs = [fig1_1, fig2_1]
    datas = [df_glb_ptc, df_loc_ptc]
    xlabels = ['', '']
    ylabels = [r'D (nm$^2$/s)', 'Normalized D (a.u)']
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

    # """
	# ~~~~Plot alpha~~~~
	# """
    figs = [fig1_2, fig2_2]
    datas = [df_glb_ptc, df_loc_ptc]
    xlabels = ['', '']
    ylabels = ['alpha', 'alpha']
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


    # """
	# ~~~~Plot speed~~~~
	# """
    figs = [fig3_1]
    datas = [df_glb]
    xlabels = ['Normalized distance to base']
    ylabels = ['Speed (a.u)']
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

    # """
	# ~~~~Plot speed~~~~
	# """
    figs = [fig3_2, fig3_3, fig3_4]
    datas = [df_glb_tip, df_glb_middle, df_glb_base]
    xlabels = ['', '', '']
    ylabels = ['Speed at tip (a.u)', 'Speed at middle (a.u)',
                'Speed at base (a.u)']
    swarmsize = [3, 1, 3]
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='exp_label',
                    y='v_norm_abs',
                    data=datas[i],
                    )
        sns.swarmplot(ax=figs[i],
                    x='exp_label',
                    y='v_norm_abs',
                    data=datas[i],
                    color=".25",
                    size=swarmsize[i],
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i],
                    cat_col='exp_label',
                    hist_col='v_norm_abs',
                    drop_duplicates=False,
                    text_pos=[0.5, 0.8],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=9,
                    horizontalalignment='center',
                    )

    # """
	# ~~~~Plot lifetime~~~~
	# """
    figs = [fig4_2, fig4_3, fig4_4]
    datas = [df_tip_lifetime, df_middle_lifetime, df_base_lifetime]
    data_col = ['tip_lifetime', 'middle_lifetime', 'base_lifetime']
    xlabels = ['', '', '']
    ylabels = ['Lifetime at tip (s)', 'Lifetime at middle (s)',
                'Lifetime at base (s)']
    swarmsize = [0.5, 0.5, 0.5]
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        sns.boxplot(ax=figs[i],
                    x='exp_label',
                    y=data_col[i],
                    data=datas[i],
                    linewidth=2,
                    )
        sns.swarmplot(ax=figs[i],
                    x='exp_label',
                    y=data_col[i],
                    data=datas[i],
                    color="1",
                    size=swarmsize[i],
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )
        add_t_test(figs[i],
                    blobs_df=datas[i],
                    cat_col='exp_label',
                    hist_col=data_col[i],
                    drop_duplicates=False,
                    text_pos=[0.5, 0.8],
                    color=(0,0,0,1),
                    fontname='Liberation Sans',
                    fontweight=9,
                    fontsize=9,
                    horizontalalignment='center',
                    )


    # """
	# ~~~~format figures~~~~
	# """
    # figs = [fig1_1, fig2_1]
    # y_max = [5000, 150000]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             yscale=[0, y_max[i]],
    #             )

    for figure in [fig1_1, fig1_2, fig2_1, fig2_2, fig3_1, fig3_2, fig3_3,
                    fig3_4, fig4_2, fig4_3, fig4_4]:
        format_spine(figure, spine_linewidth=0.5)
        format_tick(figure, tk_width=0.5)
        format_tklabel(figure, tklabel_fontsize=8)
        format_label(figure, label_fontsize=9)

    for figure in [fig3_1]:
        format_legend(figure,
                show_legend=True,
                legend_loc='upper center',
                legend_fontweight=6,
                legend_fontsize=6,
                )
















    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
