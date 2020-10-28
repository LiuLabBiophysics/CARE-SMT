from matplotlib_scalebar.scalebar import ScaleBar

def add_scalebar(ax, pixel_size, units='um',
                sb_color=(0.5, 0.5, 0.5),
                sb_pos='upper right',
                length_fraction=None,
                height_fraction=None,
                box_color=None,
                box_alpha=0,
                fontname='Arial',
                fontsize='large'
                ):
    """
    Add colorbar outside of the ax.
    Colorbar data are contained in the df.

    Parameters
    ----------
    ax : object
        matplotlib axis to add colorbar.

    pixel_size : float
		Pixel size.

    units : string
        Pixel unit. eg. 'um'

    sb_color : tuple
        RGB or RGBA tuple to define the scalebar color

    sb_pos : string
        position string. ('upper right'...)

    length_fraction : float
        scalebar length relative to ax

    height_fraction : float
        scalebar height relative to ax

    box_color : tuple
        RGB or RGBA tuple to define the scalebar background box color

    box_alpha : float
        Define the transparency of the background box color

    fontname: string
        Font used in the figure. Default is 'Arial'

    fontsize : string or integer
        Define the label text size. eg. 'large', 'x-large', 10, 40...

    Returns
    -------
    Insert a scalebar in ax.
    """

    scalebar = ScaleBar(pixel_size, units,
                    color=sb_color,
                    location=sb_pos,
                    box_color=box_color,
                    box_alpha=box_alpha,
                    length_fraction=length_fraction,
                    height_fraction=height_fraction,
                    font_properties={'family' : fontname,
                                    'size' : fontsize}
                    )

    ax.add_artist(scalebar)
