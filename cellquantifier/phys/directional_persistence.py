import pandas as pd
import numpy as np
import trackpy as tp
import math
from ..smt.track import get_d_values

def add_directional_persistence(df,
    window_width=5,
    ):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['angle', 'delta_angle', 'cos_theta', 'dir_pers']:
        if col in df:
            df = df.drop(col, axis=1)

    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    angle = (delta_y / delta_x).apply(math.atan)
    df['angle'] = angle.to_numpy()
    delta_angle = (df.groupby('particle')['angle'].apply(pd.Series.diff))
    df['delta_angle'] = delta_angle.to_numpy()
    df['cos_theta'] = df['delta_angle'].apply(math.cos)

    # """
	# ~~~~Iterate df by particle~~~~
	# """
    half_window = window_width // 2
    particles = sorted(df['particle'].unique())

    ind = 1
    tot = len(particles)
    for particle in particles:
        print("Calculating dir_pers (%d/%d)" % (ind, tot))
        ind = ind + 1

        curr_df = df[ df['particle']==particle ]
        curr_df = curr_df.sort_values(['particle', 'frame'])
        curr_df = curr_df.reset_index(drop=True)
        for index in curr_df.index:
            if half_window < (index+1) <= (len(curr_df)-half_window):

                tmp_df = curr_df.iloc[index-half_window:index+half_window+1, :]

                curr_df.loc[index, 'dir_pers'] = tmp_df['cos_theta'].mean()

            else:
                curr_df.loc[index, 'dir_pers'] = None

        df.loc[df['particle']==particle, 'dir_pers'] = curr_df['dir_pers'].to_numpy()

    return df
