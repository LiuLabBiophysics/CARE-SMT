import numpy as np

def compute_lin_vel(df, pixel_size=1, frame_rate=1):

    """
    Compute the instantaneous linear velocity of each particle

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x,y columns

    pixel_size: float, optional
        Size of each pixel in micrometers

    frame_rate: float, optional
        Acquisition frequency in frames per second

    Returns
    -------
    df: DataFrame
        DataFrame with added linear velocity columns

    Examples
    --------
    >>>import pandas as pd
    >>>from cellquantifier.phys import compute_lin_vel
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = compute_lin_vel(df)
    >>>print(df[['particle', 'lin_vel_mag', 'lin_vel_x', 'lin_vel_y']])

    """

    df = df.sort_values(['particle', 'frame'])
    df['lin_vel_x'] = pixel_size*frame_rate*df['x'].diff()
    df['lin_vel_y'] = pixel_size*frame_rate*df['y'].diff()
    df['lin_vel_mag'] = np.sqrt((df['lin_vel_x'])**2 + (df['lin_vel_y'])**2)
    df = df.fillna(0)

    return df

def compute_lin_acce(df, frame_rate=1):

    """
    Compute the instantaneous linear acceleration of each particle

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x,y columns

    frame_rate: float
        Acquisition frequency in frames per second

    Returns
    -------
    df: DataFrame
        DataFrame with added linear acceleration columns

    Examples
    --------
    >>>import pandas as pd
    >>>from cellquantifier.phys import compute_lin_accce
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = compute_lin_acce(df)
    >>>print(df[['particle', 'lin_acce_mag', 'lin_acce_x', 'lin_acce_y']])
    """

    if not 'lin_vel_mag' in df.columns:
        df = compute_lin_vel(df)

    df = df.sort_values(['particle', 'frame'])
    df['lin_acce_x'] = df['lin_vel_x'].diff()
    df['lin_acce_y'] = df['lin_vel_y'].diff()
    df['lin_acce_mag'] = frame_rate*np.sqrt((df['lin_acce_x'])**2 + (df['lin_acce_y'])**2)
    df = df.fillna(0)

    return df

def compute_force(df):

    """
    Compute the instantaneous linear force on each particle

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x,y columns

    Returns
    -------
    df: DataFrame
        DataFrame with added force columns

    Examples
    --------
    >>>import pandas as pd
    >>>from cellquantifier.phys import compute_force
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = compute_force(df)
    >>>print(df[['particle', 'force_x', 'force_y', 'force_mag']])

    """

    if not 'lin_acce_mag' in df.columns:
        df = compute_lin_acce(df)

    df = df.sort_values(['particle', 'frame'])
    df['force_x'] = df['mass']*df['lin_acce_x']
    df['force_y'] = df['mass']*df['lin_acce_y']
    df['force_mag'] = np.sqrt((df['force_x'])**2 + (df['force_y'])**2)
    df = df.fillna(0)

    return df

import numpy as np

def compute_ang_vel(df, frame_rate=1):

    """
    Compute the instantaneous angular velocity of each particle

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x,y columns

    frame_rate: float
        Acquisition frequency in frames per second

    Returns
    -------
    df: DataFrame
        DataFrame with added angular velocity columns

    Examples
    --------
    >>>import pandas as pd
    >>>from cellquantifier.phys import compute_ang_vel
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = compute_ang_vel(df)
    >>>print(df[['particle', 'ang_vel_mag']])

    """

    df = df.sort_values(['particle', 'frame'])
    df['ang_vel_mag'] = frame_rate*df['phi'].diff()
    df = df.fillna(0)

    return df

def compute_ang_acce(df, frame_rate=1):

    """
    Compute the instantaneous angular acceleration of each particle

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x,y columns

    frame_rate: float
        Acquisition frequency in frames per second

    Returns
    -------
    df: DataFrame
        DataFrame with added angular acceleration columns

    Examples
    --------
    >>>import pandas as pd
    >>>from cellquantifier.phys import compute_ang_acce
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = compute_ang_acce(df)
    >>>print(df[['particle', 'ang_acce_mag']])

    """

    if not 'ang_vel_mag' in df.columns:
        df = compute_ang_vel(df)

    df = df.sort_values(['particle', 'frame'])
    df['ang_acce_mag'] = frame_rate*df['ang_vel_mag'].diff()
    df = df.fillna(0)

    return df
