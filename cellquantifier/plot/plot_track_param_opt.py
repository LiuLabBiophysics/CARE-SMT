import matplotlib.pyplot as plt
import seaborn as sns
from .plotutil import *

def plot_track_param_opt(

        track_param_name,
        track_param_unit,

        track_param_list,
        particle_num_list,
        df,
        mean_D_list,
        mean_alpha_list,

        ):
    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(figsize=(18, 8))

    fig1 = whole_page.inset_axes([0.13, 0.65, 0.2, 0.35])
    fig2 = whole_page.inset_axes([0.45, 0.65, 0.2, 0.35])
    fig3 = whole_page.inset_axes([0.77, 0.65, 0.2, 0.35])
    fig4 = whole_page.inset_axes([0.13, 0.15, 0.2, 0.35])
    fig5 = whole_page.inset_axes([0.45, 0.15, 0.2, 0.35])
    fig6 = whole_page.inset_axes([0.77, 0.15, 0.2, 0.35])

    for axis in [whole_page]:
        axis.set_xticks([]); axis.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)


    # """
	# ~~~~Plot fig2~~~~
	# """
    if track_param_unit:
        xlabel = track_param_name + ' (' + track_param_unit + ')'
    else:
        xlabel = track_param_name

    fig1.plot(track_param_list, particle_num_list, 'o-')
    set_xylabel(fig1,
                xlabel=xlabel,
                ylabel='total particle number',
                )

    sns.boxplot(x=track_param_name,
                y="D",
                data=df,
                ax=fig2,
                )
    set_xylabel(fig2,
                xlabel=xlabel,
                ylabel=r'D (nm$^2$/s)',
                )

    sns.boxplot(x=track_param_name,
                y="alpha",
                data=df,
                ax=fig3,
                )
    set_xylabel(fig3,
                xlabel=xlabel,
                ylabel='alpha',
                )

    fig4.plot(track_param_list, particle_num_list, 'o-')
    fig4.set_yscale('log')
    set_xylabel(fig4,
                xlabel=xlabel,
                ylabel='total particle number',
                )

    fig5.plot(track_param_list, mean_D_list, 'o-')
    set_xylabel(fig5,
                xlabel=xlabel,
                ylabel=r'mean D (nm$^2$/s)',
                )

    fig6.plot(track_param_list, mean_alpha_list, 'o-')
    set_xylabel(fig6,
                xlabel=xlabel,
                ylabel='mean alpha',
                )

    # """
	# ~~~~format figures~~~~
	# """
    figs = [fig2, fig3]
    datas = ['D', 'alpha']
    for i in range(len(figs)):
        format_scale(figs[i],
                yscale=[df[datas[i]].quantile(0.1), df[datas[i]].quantile(0.99)],
                )

    for figure in [fig1, fig2, fig3, fig4, fig5, fig6]:
        format_spine(figure, spine_linewidth=1)
        format_tklabel(figure, tklabel_fontsize=12)
        format_label(figure, label_fontsize=12)

    return fig
