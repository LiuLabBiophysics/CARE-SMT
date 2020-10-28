import pandas as pd
import matplotlib.pyplot as plt
# from ..plot.plotutil import format_ax

def show_train_stats(path):

    df = pd.read_csv(path)
    fig,ax = plt.subplots(2,3)


    # """
    # ~~~~~~~~~~bg-recall~~~~~~~~~~~~~~
    # """

    ax[0,0].plot(df['val_background_recall'], color='red', label='Validation')
    ax[0,0].plot(df['background_recall'], color='blue', label='Train')
    ax[0,0].set_title(r'$\mathbf{Background}$', fontsize=10)
    format_ax(ax[0,0], ylabel=r'$\mathbf{Recall}$', ax_is_box=False, yscale=[0,1,None,None])
    ax[0,0].legend(loc='lower right')

    # """
    # ~~~~~~~~~~int-recall~~~~~~~~~~~~~~
    # """

    ax[0,1].plot(df['val_interior_recall'], color='red', label='Validation')
    ax[0,1].plot(df['interior_recall'], color='blue', label='Train')
    ax[0,1].set_title(r'$\mathbf{Interior}$', fontsize=10)
    format_ax(ax[0,1], ax_is_box=False, yscale=[0,1,None,None])
    ax[0,1].legend(loc='lower right')

    # """
    # ~~~~~~~~~~boundary-recall~~~~~~~~~~~~~~
    # """

    ax[0,2].plot(df['val_boundary_recall'], color='red', label='Validation')
    ax[0,2].plot(df['boundary_recall'], color='blue', label='Train')
    ax[0,2].set_title(r'$\mathbf{Boundary}$', fontsize=10)
    format_ax(ax[0,2], ax_is_box=False, yscale=[0,1,None,None])
    ax[0,2].legend(loc='lower right')
    # """
    # ~~~~~~~~~~bg-precision~~~~~~~~~~~~~~
    # """

    ax[1,0].plot(df['val_background_precision'], color='red', label='Validation')
    ax[1,0].plot(df['background_precision'], color='blue', label='Train')
    format_ax(ax[1,0], ylabel=r'$\mathbf{Precision}$', ax_is_box=False, yscale=[0,1,None,None])
    ax[1,0].legend(loc='lower right')

    # """
    # ~~~~~~~~~~int-precision~~~~~~~~~~~~~~
    # """

    ax[1,1].plot(df['val_interior_precision'], color='red', label='Validation')
    ax[1,1].plot(df['interior_precision'], color='blue', label='Train')
    format_ax(ax[1,1], ax_is_box=False, yscale=[0,1,None,None])
    ax[1,1].legend(loc='lower right')

    # """
    # ~~~~~~~~~~boundary-precision~~~~~~~~~~~~~~
    # """

    ax[1,2].plot(df['val_boundary_precision'], color='red', label='Validation')
    ax[1,2].plot(df['boundary_precision'], color='blue', label='Train')
    format_ax(ax[1,2], ax_is_box=False, yscale=[0,1,None,None])
    ax[1,2].legend(loc='lower right')

    plt.tight_layout()
    plt.show()
