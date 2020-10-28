import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *


def fig_quick_antigen_3(df=pd.DataFrame([])):
    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty:
        # """
        # ~~~~~~~~traj_length filter~~~~~~~~
        # """
        if 'traj_length' in df:
            df = df[ df['traj_length']>=20 ]

        # """
        # ~~~~~~~~travel_dist filter~~~~~~~~
        # """
        if 'travel_dist' in df:
            travel_dist_min = 0
            travel_dist_max = 7
            df = df[ (df['travel_dist']>=travel_dist_min) & \
            					(df['travel_dist']<=travel_dist_max) ]

        # """
        # ~~~~~~~~add particle type filter~~~~~~~~
        # """
        if 'particle_type' in df:
        	df = df[ df['particle_type']!='--none--']




        df['date'] = df['raw_data'].astype(str).str[0:6]
        df_mal = df[ df['date'].isin(['200205']) ]
        df_mal_A = df_mal[ df_mal['particle_type']=='A' ]
        df_mal_B = df_mal[ df_mal['particle_type']=='B' ]
        df_cas9 = df[ df['date'].isin(['200220']) ]
        df_cas9_A = df_cas9[ df_cas9['particle_type']=='A' ]
        df_cas9_B = df_cas9[ df_cas9['particle_type']=='B' ]

        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')

        df_mal_particle = df_mal.drop_duplicates('particle')
        df_mal_A_particle = df_mal_A.drop_duplicates('particle')
        df_mal_B_particle = df_mal_B.drop_duplicates('particle')

        df_cas9_particle = df_cas9.drop_duplicates('particle')
        df_cas9_A_particle = df_cas9_A.drop_duplicates('particle')
        df_cas9_B_particle = df_cas9_B.drop_duplicates('particle')




    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(17, 22))
    left_page = whole_page.inset_axes([0.025, 0, 0.45, 1])
    right_page = whole_page.inset_axes([0.525, 0, 0.45, 1])

    fig1 = left_page.inset_axes([0, 0.67, 1, 0.33])
    fig2 = left_page.inset_axes([0, 0.56, 1, 0.11])
    fig3 = left_page.inset_axes([0, 0.45, 1, 0.11])

    fig1_1 = fig1.inset_axes([0.13, 0.78, 0.2, 0.2])
    fig1_2 = fig1.inset_axes([0.45, 0.78, 0.2, 0.2])
    fig1_3 = fig1.inset_axes([0.77, 0.78, 0.2, 0.2])
    fig1_4 = fig1.inset_axes([0.13, 0.48, 0.2, 0.2])
    fig1_5 = fig1.inset_axes([0.45, 0.48, 0.2, 0.2])
    fig1_6 = fig1.inset_axes([0.77, 0.48, 0.2, 0.2])
    fig1_7 = fig1.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig1_8 = fig1.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig1_9 = fig1.inset_axes([0.77, 0.18, 0.2, 0.2])

    fig2_1 = fig2.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig2_2 = fig2.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig3_1 = fig3.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig3_2 = fig3.inset_axes([0.6, 0.35, 0.3, 0.6])



    fig4 = right_page.inset_axes([0, 0.67, 1, 0.33])
    fig5 = right_page.inset_axes([0, 0.56, 1, 0.11])
    fig6 = right_page.inset_axes([0, 0.45, 1, 0.11])

    fig4_1 = fig4.inset_axes([0.13, 0.78, 0.2, 0.2])
    fig4_2 = fig4.inset_axes([0.45, 0.78, 0.2, 0.2])
    fig4_3 = fig4.inset_axes([0.77, 0.78, 0.2, 0.2])
    fig4_4 = fig4.inset_axes([0.13, 0.48, 0.2, 0.2])
    fig4_5 = fig4.inset_axes([0.45, 0.48, 0.2, 0.2])
    fig4_6 = fig4.inset_axes([0.77, 0.48, 0.2, 0.2])
    fig4_7 = fig4.inset_axes([0.13, 0.18, 0.2, 0.2])
    fig4_8 = fig4.inset_axes([0.45, 0.18, 0.2, 0.2])
    fig4_9 = fig4.inset_axes([0.77, 0.18, 0.2, 0.2])

    fig5_1 = fig5.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig5_2 = fig5.inset_axes([0.6, 0.35, 0.3, 0.6])

    fig6_1 = fig6.inset_axes([0.13, 0.35, 0.3, 0.6])
    fig6_2 = fig6.inset_axes([0.6, 0.35, 0.3, 0.6])



    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [left_page, right_page, whole_page,
                fig1, fig2, fig3, fig4, fig5, fig6]:
        axis.set_xticks([]); axis.set_yticks([])

    # # """
	# # ~~~~Plot mean_msd~~~~
	# # """
    # figs = [fig1_1, fig1_4, fig1_7, fig4_1, fig4_4, fig4_7]
    # datas = [df_mal, df_mal_A, df_mal_B, df_cas9, df_cas9_A, df_cas9_B]
    # for i in range(len(figs)):
    #     print("\n")
    #     print(figs[i].name)
    # 	# print("Ploting: " + str(figs[i]))
    #     add_mean_msd(figs[i], datas[i],
    #                 cat_col='exp_label',
    #                 # cat_order=['MalKN', 'WT', 'MalOE'],
    #                 pixel_size=0.163,
    #                 frame_rate=2,
    #                 divide_num=5,
    #                 RGBA_alpha=0.5,
    #                 fitting_linewidth=1.5,
    #                 elinewidth=0.5,
    #                 markersize=4,
    #                 capsize=1,
    #                 set_format=False)
    #     set_xylabel(figs[i],
    #                 xlabel='Time (s)',
    #                 ylabel=r'MSD (nm$^2$)',
    #                 )

    # # """
	# # ~~~~Plot D~~~~
	# # """
    # figs = [fig1_2, fig1_5, fig1_8, fig4_2, fig4_5, fig4_8]
    # datas = [df_mal_particle, df_mal_A_particle, df_mal_B_particle,
    #         df_cas9_particle, df_cas9_A_particle, df_cas9_B_particle]
    # xlabels = ['Particle All', 'Particle A', 'Particle B',
    #         'Particle All', 'Particle A', 'Particle B']
    # for i in range(len(figs)):
    #     print("\n")
    #     print(figs[i].name)
    # 	# print("Ploting: " + st
    #     sns.boxplot(ax=figs[i],
    #                 x='exp_label',
    #                 y='D',
    #                 data=datas[i],
    #                 fliersize=2,
    #                 )
    #     set_xylabel(figs[i],
    #                 xlabel=xlabels[i],
    #                 ylabel=r'D (nm$^2$/s)',
    #                 )
    #
    # # """
	# # ~~~~Plot alpha~~~~
	# # """
    # figs = [fig1_3, fig1_6, fig1_9, fig4_3, fig4_6, fig4_9]
    # datas = [df_mal_particle, df_mal_A_particle, df_mal_B_particle,
    #         df_cas9_particle, df_cas9_A_particle, df_cas9_B_particle]
    # xlabels = ['Particle All', 'Particle A', 'Particle B',
    #         'Particle All', 'Particle A', 'Particle B']
    # for i in range(len(figs)):
    #     print("\n")
    #     print(figs[i].name)
    # 	# print("Ploting: " + st
    #     sns.boxplot(ax=figs[i],
    #                 x='exp_label',
    #                 y='alpha',
    #                 data=datas[i],
    #                 fliersize=2,
    #                 )
    #     set_xylabel(figs[i],
    #                 xlabel=xlabels[i],
    #                 ylabel='alpha',
    #                 )
    #
    #
    # # """
	# # ~~~~Plot lifetime~~~~
	# # """
    # figs = [fig2_1, fig2_2, fig5_1, fig5_2]
    # datas = [df_mal_A_particle, df_mal_B_particle,
    #         df_cas9_A_particle, df_cas9_B_particle]
    # xlabels = ['Particle A', 'Particle B',
    #         'Particle A', 'Particle B']
    # for i in range(len(figs)):
    #     print("\n")
    #     print(figs[i].name)
    # 	# print("Ploting: " + st
    #     sns.boxplot(ax=figs[i],
    #                 x='exp_label',
    #                 y='lifetime',
    #                 data=datas[i],
    #                 fliersize=2,
    #                 )
    #     set_xylabel(figs[i],
    #                 xlabel=xlabels[i],
    #                 ylabel='lifetime (frame)',
    #                 )
    #

    # """
	# ~~~~format figures~~~~
	# """
    # figs = [fig1_2, fig1_5, fig1_8, fig4_2, fig4_5, fig4_8]
    # y_max = [15000, 15000, 15000, 10000, 10000, 10000]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             yscale=[0, y_max[i]],
    #             )
    #
    # # figs = [fig1_3, fig1_6, fig1_9, fig4_3, fig4_6, fig4_9]
    # # for i in range(len(figs)):
    # #     format_scale(figs[i],
    # #             yscale=[df['alpha'].quantile(0.1), df['alpha'].quantile(0.99)],
    # #             )
    #
    # figs = [fig2_1, fig2_2, fig5_1, fig5_2]
    # y_max = [250, 310, 350, 350]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             yscale=[0, y_max[i]],
    #             )

    for figure in [fig1_1, fig1_2, fig1_3, fig1_4, fig1_5, fig1_6,
        fig1_7, fig1_8, fig1_9, fig2_1, fig2_2, fig3_1, fig3_2,
        fig4_1, fig4_2, fig4_3, fig4_4, fig4_5, fig4_6, fig4_7,
        fig4_8, fig4_9, fig5_1, fig5_2, fig6_1, fig6_2]:
        format_spine(figure, spine_linewidth=1)
        format_tklabel(figure, tklabel_fontsize=8)
        format_label(figure, label_fontsize=9)

    # for figure in [fig1_1, fig1_4, fig1_7, fig4_1, fig4_4, fig4_7]:
    #     format_legend(figure,
    #             show_legend=True,
    #             legend_loc='upper left',
    #             legend_fontweight=6,
    #             legend_fontsize=6,
    #             )
















    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
