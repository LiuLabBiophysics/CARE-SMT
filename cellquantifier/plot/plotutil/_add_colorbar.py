import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


def add_outside_colorbar(ax, df, data_col='D',
                        cb_pos='right',
                        cb_size='5%',
                        cb_pad=0.1,
                        cb_colormap='coolwarm',
                        cb_orientation='vertical',
                        cb_extend='both',
                        cb_min=None,
                        cb_max=None,
                        cb_major_ticker=None,
                        cb_minor_ticker=None,
                        cb_tick_loc='right',
                        show_colorbar=True,
                        label_str=r'D (nm$^2$/s)',
                        label_font_size=20,
                        label_color=(0, 0, 0),
                        label_font='Arial',
                        label_weight='bold',
                        ):
    """
    Add colorbar outside of the ax.
    Colorbar data are contained in the df.

    Parameters
    ----------
    ax : object
        matplotlib axis to add colorbar.

    df : DataFrame
		DataFrame containing 'D' columns

    data_col : string
        Data colunm name in the df

    label_str : string
        Use this as the colorbar label

    label_font_size : string or integer
        Define the label text size. eg. 'large', 'x-large', 10, 40...

    label_color : tuple
        RGBA or RGB tuple to define the label and tick color

    label_font: string
        Font used in the figure. Default is 'Arial'

    label_weight : string
        font weight string as mpl.ax.text. ('light', 'bold'...)

    cb_pos : string
        append_axes position string. ('right', 'top'...)

    cb_size : string
        colorbar size string. ('5%')

    cb_pad : float
        gap between colorbar and ax in inches.

    cb_colormap : string

    cb_orientation : string
        eg. 'vertical', 'horizontal'

    cb_extend : string
        eg. 'neither', 'both', None...

    cb_min, cb_max: float
        [cb_min, cb_max] is the color bar range.

    cb_major_ticker, cb_minor_ticker: float
        Major and minor setting for the color bar

    show_colorbar : bool
        If False, only return colormap and df, no colorbar added.

    Returns
    -------
    Annotate colorbar outside of the ax.
    df, colormap returned.
    """


    colormap = plt.cm.get_cmap(cb_colormap)

    # """
    # ~~~~~~~~~~~customized the tick; or automated~~~~~~~~~~~~~~
    # """
    norm_col = data_col + '_norm'
    if cb_max!=None and cb_min!=None \
    and cb_major_ticker!=None and cb_minor_ticker!=None:
        df[norm_col] = (df[data_col] - cb_min)/(cb_max - cb_min)
        norm = mpl.colors.Normalize(vmin = cb_min, vmax = cb_max)
    else:
        df[norm_col] = (df[data_col] - df[data_col].min()) \
                        / (df[data_col].max() - df[data_col].min())
        norm = mpl.colors.Normalize(vmin = df[data_col].min(),
                                    vmax = df[data_col].max())

    if show_colorbar:
        # """
        # ~~~~~~~~~~~Add D value scale bar to left plot~~~~~~~~~~~~~~
        # """
        # backup code for add_inside_colorbar()
        #ax_cb = ax.inset_axes([0.1, 0.85, 0.4, 0.03])

        # Create an axes (ax_cb) on the right side
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes(cb_pos, size=cb_size, pad=cb_pad)


        cb1 = mpl.colorbar.ColorbarBase(ax_cb,
                                cmap=colormap,
                                norm=norm,
                                orientation=cb_orientation,
                                extend=cb_extend,
                                ticklocation=cb_tick_loc)

        # Setup colorbar format
        cb1.outline.set_visible(False)
        plt.yticks(fontname=label_font,
                    weight=label_weight,
                    color=label_color,
                    size=label_font_size)

        # Setup colorbar label
        ax_cb.set_ylabel(label_str,
                        fontname=label_font,
                        color=label_color,
                        weight=label_weight,
                        size=label_font_size)

        # customize colorbar scale if needed
        if cb_max!=None and cb_min!=None \
        and cb_major_ticker!=None and cb_minor_ticker!=None:
            ax_cb.yaxis.set_major_locator(ticker.MultipleLocator(cb_major_ticker))
            ax_cb.yaxis.set_minor_locator(ticker.MultipleLocator(cb_minor_ticker))

    return df, colormap
