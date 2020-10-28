from .plotutil import *
from datetime import date

def plot_phys_1(blobs_df,
                cat_col,
                pixel_size,
                frame_rate,
                divide_num,
                RGBA_alpha=0.5,
                do_gmm=False):

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    add_mean_msd(ax[0], blobs_df,
                cat_col=cat_col,
                pixel_size=pixel_size,
                frame_rate=frame_rate,
                divide_num=divide_num,
                RGBA_alpha=RGBA_alpha
                )
    ax[0].legend(loc='upper left', frameon=False, fontsize=13)



    add_D_hist(ax[1], blobs_df,
                cat_col=cat_col,
                RGBA_alpha=RGBA_alpha)
    add_t_test(ax[1], blobs_df,
                cat_col=cat_col,
                hist_col=['D'],
                fontsize=20)
    if do_gmm:
        add_gmm(ax[1], blobs_df,
                cat_col=cat_col,
                n_comp=3,
                hist_col='D',
                RGBA_alpha=RGBA_alpha)
    ax[1].legend(loc='upper right', frameon=False, fontsize=13)




    add_alpha_hist(ax[2],
                blobs_df,
                cat_col=cat_col,
                RGBA_alpha=RGBA_alpha)
    add_t_test(ax[2], blobs_df,
                cat_col=cat_col,
                hist_col=['alpha'],
                fontsize=20)
    if do_gmm:
        add_gmm(ax[2], blobs_df,
                cat_col=cat_col,
                n_comp=1,
                hist_col='alpha',
                RGBA_alpha=RGBA_alpha)
    ax[2].legend(loc='upper right', frameon=False, fontsize=13)


    plt.tight_layout()
    plt.show()

    # # """
    # # ~~~~~~~~~~~Save the plot as pdf, and open the pdf in browser~~~~~~~~~~~~~~
    # # """
    # start_ind = root_name.find('_')
    # end_ind = root_name.find('_', start_ind+1)
    # today = str(date.today().strftime("%y%m%d"))
    # fig.savefig(output_path + today + root_name[start_ind:end_ind] + '-mergedResults.pdf')
    # plt.show()

    return fig
