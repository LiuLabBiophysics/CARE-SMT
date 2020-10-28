import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from ..smt import get_d_values
from scipy.stats import norm, expon
import trackpy as tp
import numpy as np
from ..qmath import *
import pandas as pd


def fig_quick_msd(
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
    if not df.empty:
        pixel_size = df['pixel_size'].mean()
        frame_rate = df['frame_rate'].mean()
        divide_num = 3

        im = tp.imsd(df,
                    mpp=pixel_size,
                    fps=frame_rate,
                    max_lagtime=np.inf,
                    )

        n = int(round(len(im.index)/divide_num))
        im = im.head(n)

        # Remove NaN, Remove non-positive value before calculate log()
        msd = im.dropna()
        msd = msd[msd > 0]

        if len(msd) > 2: # Only fit when msd has more than 2 data points
            x = msd.index.values
            y = msd.to_numpy()
            y = y*1e6 #convert um^2 to nm^2
            print(y[0])
            # y = y - y[0]

            try:
                popt_msd1 = fit_msd1_log(x, y)
            except:
                popt_msd1 = (0, 0)

            y_msd1 = msd1(x, popt_msd1[0], popt_msd1[1])

    # """
	# ~~~~Plot D~~~~
	# """
    figs = [fig1]
    datas = [df]
    xlabels = ['Time (s)']
    ylabels = [r'MSD (nm$^2$)']
    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        figs[i].plot(x, y, '-o', color=(0,0,0,0.8), linewidth=2, markersize=10)
        figs[i].plot(x, y_msd1, '--', color=(0,0,0,0.6), linewidth=5)
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

    # figs = [fig1]
    # xscales = [[1.5, 20.5, 5], ]
    # yscales = [[15000, 48000, 10000], ]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             xscale=xscales[i],
    #             yscale=yscales[i],
    #             )





    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.tiff', dpi=300)
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()
