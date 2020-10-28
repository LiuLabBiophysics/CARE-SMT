import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from ..smt import get_d_values
from scipy.stats import norm, expon
import numpy as np
from ..qmath import t_test
import pandas as pd


def fig_quick_osc(
    df=pd.DataFrame([]),
    ):
    # """
	# ~~~~~~~~~~~Initialize the page layout~~~~~~~~~~~~~~
	# """
    fig, whole_page = plt.subplots(1, 1, figsize=(8, 6))
    fig1 = whole_page.inset_axes([0.25, 0.25, 0.7, 0.7])

    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    for axis in [whole_page]:
        axis.set_xticks([]); axis.set_yticks([])


    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    if not df.empty :
        pass


    # """
	# ~~~~Plot D~~~~
	# """
    figs = [fig1]
    datas = [df]
    xlabels = ['Frame']
    ylabels = ['Normalized distance to base (a.u)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))
        figs[i].plot(
                    datas[i]['frame'], datas[i]['h_norm'], '-o',
                    color=(0,0,0),
                    linewidth=1,
                    markersize=5
                    )
        set_xylabel(figs[i],
                    xlabel=xlabels[i],
                    ylabel=ylabels[i],
                    )


    # """
	# ~~~~format figures~~~~
	# """
    for figure in [fig1,
                ]:
        format_spine(figure, spine_linewidth=1)
        format_tick(figure, tk_width=1)
        format_tklabel(figure, tklabel_fontsize=12)
        format_label(figure, label_fontsize=15)





    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.tiff', dpi=300)
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
