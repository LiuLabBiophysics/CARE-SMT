def add_foci_num(df):
    """
    Add column to single cell df: 'foci_num'
    This function does not work for mergedData.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'fittData'

    Returns
    -------
    df: DataFrame
        DataFrame with added columns
    """

    frames = df['frame'].unique()

    for frame in frames:
        curr_df = df[ df['frame']==frame ]

        foci_num = len(curr_df)
        foci_peaksum = curr_df['peak'].sum()
        foci_peakmean = curr_df['peak'].mean()
        df.loc[df['frame']==frame, 'foci_num'] = foci_num
        df.loc[df['frame']==frame, 'foci_peaksum'] = foci_peaksum
        df.loc[df['frame']==frame, 'foci_peakmean'] = foci_peakmean

        if 'area' in curr_df:
            foci_areasum = curr_df['area'].sum()
            foci_areamean = curr_df['area'].mean()
            foci_pkareasum = (curr_df['peak'] * curr_df['area']).sum()
            df.loc[df['frame']==frame, 'foci_areasum'] = foci_areasum
            df.loc[df['frame']==frame, 'foci_areamean'] = foci_areamean
            df.loc[df['frame']==frame, 'foci_pkareasum'] = foci_pkareasum

    df['foci_num_norm'] = df['foci_num'] / df['foci_num'].max()
    # df['foci_peaksum_norm'] = df['foci_peaksum'] / df['foci_peaksum'].max()
    # df['foci_peakmean_norm'] = df['foci_peakmean'] / df['foci_peakmean'].max()
    df = df.sort_values(by='frame')
    df['foci_peaksum_norm'] = df['foci_peaksum'] / df.loc[df.index[0], 'foci_peaksum']
    df['foci_peakmean_norm'] = df['foci_peakmean'] / df.loc[df.index[0], 'foci_peakmean']

    if 'area' in df:
        df['foci_areasum_norm'] = df['foci_areasum'] / df['foci_areasum'].max()
        df['foci_areamean_norm'] = df['foci_areamean'] / df['foci_areamean'].max()
        df['foci_pkareasum_norm'] = df['foci_pkareasum'] / df['foci_pkareasum'].max()

    return df
