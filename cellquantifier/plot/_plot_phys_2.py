from .plotutil import *
import matplotlib.pyplot as plt

def plot_phys_2(phys_df, cat_col):

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    add_violin_1(ax[0],
                df=phys_df,
                data_col='A',
                cat_col='exp_label')

    add_t_test(ax[0],
                blobs_df=phys_df,
                cat_col=cat_col,
                hist_col='A',
                drop_duplicates=False,
                text_pos=[0.95, 0.8])

    ax[0].legend(loc='upper left', frameon=False, fontsize=15)
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylabel('Peak intensity (ADU)', fontsize=15, fontweight='bold')

    add_violin_1(ax[1],
                df=phys_df,
                data_col='area',
                cat_col='exp_label')

    add_t_test(ax[1],
                blobs_df=phys_df,
                cat_col=cat_col,
                hist_col='area',
                drop_duplicates=False,
                text_pos=[0.95, 0.8])

    ax[1].legend(loc='upper left', frameon=False, fontsize=15)
    ax[1].tick_params(labelsize=15)
    ax[1].set_ylabel('Foci area (pixel^2)', fontsize=15, fontweight='bold')
    ax[1].set_ylim([0, 30])

    phys_df['A/area'] = phys_df['A']/phys_df['area']
    add_violin_1(ax[2],
                df=phys_df,
                data_col='A/area',
                cat_col='exp_label')

    add_t_test(ax[2],
                blobs_df=phys_df,
                cat_col=cat_col,
                hist_col='A/area',
                drop_duplicates=False,
                text_pos=[0.95, 0.8])

    ax[2].legend(loc='upper left', frameon=False, fontsize=15)
    ax[2].tick_params(labelsize=15)
    ax[2].set_ylabel('Compactness (ADU / pixel^2)', fontsize=15, fontweight='bold')
    ax[2].set_ylim([0, 25])

    plt.tight_layout()
    plt.show()


def plot_phys_3(phys_df, cat_col):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    df_drop_duplicates = phys_df.drop_duplicates('particle')

    add_violin_1(ax[0],
                df=phys_df,
                data_col='traj_lc',
                cat_col='exp_label')

    add_t_test(ax[0],
                blobs_df=phys_df,
                cat_col=cat_col,
                hist_col='traj_lc',
                drop_duplicates=True,
                text_pos=[0.95, 0.8])

    ax[0].legend(loc='upper left', frameon=False, fontsize=15)
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylabel('Traj Radius (pixel)',
                    fontsize=15, fontweight='bold')
    ax[0].set_ylim([0.2, 1.6])

    add_violin_1(ax[1],
                df=phys_df,
                data_col='traj_area',
                cat_col='exp_label')

    add_t_test(ax[1],
                blobs_df=phys_df,
                cat_col=cat_col,
                hist_col='traj_area',
                drop_duplicates=True,
                text_pos=[0.95, 0.8])

    ax[1].legend(loc='upper left', frameon=False, fontsize=15)
    ax[1].tick_params(labelsize=15)
    ax[1].set_ylabel('Traj Area (pixel^2)',
                    fontsize=15, fontweight='bold')
    ax[1].set_ylim([0, 2.5])

    plt.tight_layout()
    plt.show()
