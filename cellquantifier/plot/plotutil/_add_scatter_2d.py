def add_scatter_2d(ax, df, axis_cols,
            data_col=None,
            color=None,
            ):
    """
    Add 2d scatter in matplotlib axis.

    Pseudo code
    ----------
    1. If df is empty, return.
    2. Add scatters in the ax.

    Parameters
    ----------
    ax : object
        matplotlib axis.

    df : DataFrame
		DataFrame containing necessary data.

    axis_cols : list of str
        Columns used to plot as the axis.
        For example, ['col1', 'col2'],
        'col1' is used to plot as the x value.
        'col2' is used to plot as the y value.

    data_col : str, optional
        Column used to plot as the size of the sactter.

    color : tuple, optional
        Color used to plot the sactter.

    Returns
    -------
    Add scatters in the ax.

    Examples
	--------
    """

    # """
    # ~~~~Check if df is empty~~~~
    # """
    if df.empty:
    	return

    # """
    # ~~~~Add scatters in the ax~~~~
    # """
    x, y = df[axis_cols[0]], df[axis_cols[1]]
    s = df[data_col] if data_col else None
    ax.scatter(x=x, y=y, s=s, c=[color])
