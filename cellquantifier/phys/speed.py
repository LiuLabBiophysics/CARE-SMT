import pandas as pd
import numpy as np

def add_speed(df):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['v']:
        if col in df:
            df = df.drop(col, axis=1)

    # """
	# ~~~~calculate 'v', the unit is px/frame~~~~
	# """
    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    df['dx'] = delta_x
    delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    df['dy'] = delta_y
    speed = (delta_x**2 + delta_y**2) ** 0.5 * df['pixel_size'] * df['frame_rate']

    # """
	# ~~~~filter out 'v' which is not adjacent~~~~
	# """
    delta_frame = (df.groupby('particle')['frame'].apply(pd.Series.diff))
    df['adjacent_frame'] = delta_frame==1
    df['v'] = speed[ df['adjacent_frame'] ]

    # """
	# ~~~~filter out 'v' which is not big_step~~~~
	# """
    particles = df['particle'].unique()
    for particle in particles:
        curr_df = df[ df['particle']==particle ]
        v_max = curr_df['v'].max()
        big_step = curr_df['v']>=v_max*0.2
        df.loc[df['particle']==particle, 'big_step'] = big_step

    df['v'] = speed[ df['adjacent_frame'] & df['big_step'] ]

    return df
