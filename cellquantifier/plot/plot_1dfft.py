import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from .plotutil import *
import seaborn as sns

def plot_1dfft(df, output_path,
    data_col='y',
    ):
    """
    Plot one dimensional fft plot for each particle.
    Save pdf file for each particle.
    (WATCH OUT!!! A LOT OF PDFS!!!)

    Pseudo code
    ----------
    1. Iterate every 'raw_data' file.
    2. Iterate every 'particle'.
    3. Plot fitting curve for each particle, and save as pdf.

    Parameters
    ----------
    df : DataFrame
        With columns ['x', 'y', 'frame', 'particle', 'raw_data',
            'pixel_size', 'frame_rate']

    output_path : str
		Output path for all the pdf files

    data_col : str, optional
        Column used to plot histogram.

    Returns
    -------
    None return values. A bunch of pdf files.
    """

    result_df = pd.DataFrame(np.array([]))
    print(result_df)

    raw_files = df['raw_data'].unique()

    ind = 1
    tot = len(raw_files)
    for raw_file in raw_files:
        print("\n")
        print("Plot msd fitting curves (%d/%d): %s" % (ind, tot, raw_file))
        ind = ind + 1

        curr_df = df[ df['raw_data']==raw_file ]
        particles = curr_df['particle'].unique()

        for particle in particles:
            curr_ptcl = curr_df[ curr_df['particle']==particle ]
            curr_ptcl = curr_ptcl.sort_values('frame')

            pixel_size = curr_ptcl['pixel_size'].mean()
            frame_rate = curr_ptcl['frame_rate'].mean()

            x = curr_ptcl['frame'] / frame_rate
            y = curr_ptcl[data_col]

            N = len(x)
            T = 1 / frame_rate
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)[1:]
            yf = np.abs( fft(y)[0:N//2] )[1:]

            print(particle)
            result_df[particle] = yf

            print(result_df)
    print(result_df.sum(axis=1))










    fig, whole_page = plt.subplots(figsize=(18, 4))
    fig1 = whole_page.inset_axes([0.13, 0.15, 0.2, 0.7])
    fig2 = whole_page.inset_axes([0.45, 0.15, 0.2, 0.7])
    fig3 = whole_page.inset_axes([0.77, 0.15, 0.2, 0.7])

    for axis in [whole_page]:
        axis.set_xticks([]); axis.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        whole_page.spines[spine].set_visible(False)

    # fig1.plot(x, y, '-o', color=(0,0,0), linewidth=1, markersize=5)
    # fig2.plot(xf, yf, '-o', color=(0,0,0), linewidth=1, markersize=5)

    fig1.plot(result_df.index, result_df.sum(axis=1), '-o', color=(0,0,0), linewidth=1, markersize=5)






    # fig2.plot(x, y_msd1_log, '-', color=(0,0,1), linewidth=2)
    # fig3.plot(x, y, '-o', color=(0,0,0), linewidth=1, markersize=5)
    # fig3.plot(x, y_msd2, '-', color=(0,0,1), linewidth=2)
    #
    # fig1.text(-0.3,
    #         1.05,
    #         r"""
    #         MSD = 4Dt$^\alpha$
    #         D: %.2f
    #         $\alpha$: %.2f
    #         rmse: %.2f
    #         """ %(popt_msd1[0], popt_msd1[1], rmse_msd1),
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         fontname='Liberation Sans',
    #         fontsize = 12,
    #         fontweight = 'bold',
    #         color = (0,0,0),
    #         transform=fig1.transAxes,
    #         )

    # fig2.text(-0.3,
    #         1.05,
    #         r"""
    #         MSD = 4D + $\alpha$ln(t)
    #         D: %.2f
    #         $\alpha$: %.2f
    #         rmse: %.2f
    #         """ %(popt_msd1_log[0], popt_msd1_log[1], rmse_msd1_log),
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         fontname='Liberation Sans',
    #         fontsize = 12,
    #         fontweight = 'bold',
    #         color = (0,0,0),
    #         transform=fig2.transAxes,
    #         )
    #
    # fig3.text(-0.3,
    #         1.05,
    #         r"""
    #         MSD = 4Dt$^\alpha$ + c
    #         D: %.2f
    #         $\alpha$: %.2f
    #         c: %.2f
    #         rmse: %.2f
    #         """ %(popt_msd2[0], popt_msd2[1], popt_msd2[2], rmse_msd2),
    #         horizontalalignment='left',
    #         verticalalignment='top',
    #         fontname='Liberation Sans',
    #         fontsize = 12,
    #         fontweight = 'bold',
    #         color = (0,0,0),
    #         transform=fig3.transAxes,
    #         )
    #
    # whole_page.text(0,
    #         -0.15,
    #         """
    #         (MSD curve fitting model comparison)
    #         """,
    #         horizontalalignment='left',
    #         verticalalignment='bottom',
    #         fontname='Liberation Sans',
    #         fontsize = 12,
    #         fontweight = 'normal',
    #         color = (0,0,0),
    #         transform=whole_page.transAxes,
    #         )

    # """
    # ~~~~format figures~~~~
    # """
    for figure in [fig1, fig2, fig3]:
        format_spine(figure, spine_linewidth=1)
        format_tklabel(figure, tklabel_fontsize=12)
        format_label(figure, label_fontsize=12)

    file_name = output_path + raw_file[:-len('physData.csv')]
    file_name = file_name + 'particle' + str(particle) + '-1dfft.pdf'

    fig.savefig(file_name)
    plt.clf(); plt.close()
