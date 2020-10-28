import pandas as pd
import numpy as np

def add_travel_dist(df):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['travel_dist']:
        if col in df:
            df = df.drop(col, axis=1)

    # # """
	# # ~~~~add 'v', the unit is px/frame~~~~
	# # """
    # delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    # delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    # df['v'] = (delta_x**2 + delta_y**2) ** 0.5 * df['pixel_size'] * df['frame_rate']

    # """
	# ~~~~Iterate df by particle~~~~
	# """
    particles = sorted(df['particle'].unique())
    for particle in particles:
        curr_df = df[ df['particle']==particle ]

        # # """
    	# # ~~~~add 'v_max'~~~~
    	# # """
        # v_max = curr_df['v'].max()
        # df.loc[df['particle']==particle, 'v_max'] = v_max

        # """
    	# ~~~~add 'travel_dist'~~~~
    	# """
        travel_dist = ((curr_df['x'].max() - curr_df['x'].min())**2 + \
                    (curr_df['y'].max() - curr_df['y'].min())**2) ** 0.5
        df.loc[df['particle']==particle, 'travel_dist'] = travel_dist * \
                    curr_df['pixel_size'].mean()

    return df
