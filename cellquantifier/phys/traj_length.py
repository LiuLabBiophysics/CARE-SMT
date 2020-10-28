from sklearn import mixture
import numpy as np

def add_traj_length(physdf):
    """
    Add column to physdf: 'traj_length'

    Parameters
    ----------
    physdf : DataFrame
        DataFrame containing 'x', 'y', 'frame', 'particle'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'traj_length' column
    """

    particles = physdf['particle'].unique()

    for particle in particles:
        traj_length = len(physdf[ physdf['particle']==particle ])
        physdf.loc[physdf['particle']==particle, 'traj_length'] = traj_length

    return physdf
