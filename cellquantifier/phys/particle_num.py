def add_particle_num(df):
    """
    Add column to df: 'particle_num'

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'raw_data', 'particle'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'particle_num' column
    """

    raw_datas = df['raw_data'].unique()
    dfp = df.drop_duplicates('particle')

    for raw_data in raw_datas:
        particle_num = len(dfp[ dfp['raw_data']==raw_data ])
        df.loc[df['raw_data']==raw_data, 'particle_num'] = particle_num

    return df
