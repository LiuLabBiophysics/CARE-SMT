import pandas as pd
import numpy as np
import trackpy as tp
from ..smt.track import get_d_values

def add_local_D_alpha(df, pixel_size, frame_rate,
    divide_num=5,
    window_width=20,
    ):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['local_D', 'local_alpha', 'local_v']:
        if col in df:
            df = df.drop(col, axis=1)

    # """
	# ~~~~Iterate df by particle~~~~
	# """
    half_window = window_width // 2
    particles = sorted(df['particle'].unique())

    ind = 1
    tot = len(particles)
    for particle in particles:
        print("Calculating local_D and local_alpha (%d/%d)" % (ind, tot))
        ind = ind + 1

        curr_df = df[ df['particle']==particle ]
        curr_df = curr_df.sort_values(['particle', 'frame'])
        curr_df = curr_df.reset_index(drop=True)
        for index in curr_df.index:
            if half_window < (index+1) <= (len(curr_df)-half_window):

                tmp_df = curr_df.iloc[index-half_window:index+half_window+1, :]

                tmp_df_cut = tmp_df[['frame', 'x', 'y', 'particle']]
                im = tp.imsd(tmp_df_cut, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)
                tmp_df = get_d_values(tmp_df, im, divide_num)

                curr_df.loc[index, 'local_D'] = tmp_df['D'].mean()
                curr_df.loc[index, 'local_v'] = (tmp_df['D'].mean() * 4) ** 0.5
                curr_df.loc[index, 'local_alpha'] = tmp_df['alpha'].mean()

            else:
                curr_df.loc[index, 'local_D'] = None
                curr_df.loc[index, 'local_v'] = None
                curr_df.loc[index, 'local_alpha'] = None

        df.loc[df['particle']==particle, 'local_D'] = curr_df['local_D'].to_numpy()
        df.loc[df['particle']==particle, 'local_v'] = curr_df['local_v'].to_numpy()
        df.loc[df['particle']==particle, 'local_alpha'] = curr_df['local_alpha'].to_numpy()

    return df
