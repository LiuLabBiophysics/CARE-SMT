import pandas as pd
import numpy as np

def add_antigen_data(df, sorters=None):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['v', 'v_max', 'travel_dist', 'lifetime',
                'boundary_type', 'particle_type']:
        if col in df:
            df = df.drop(col, axis=1)

    # """
	# ~~~~add 'v', the unit is px/frame~~~~
	# """
    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    df['v'] = (delta_x**2 + delta_y**2) ** 0.5

    # """
	# ~~~~Iterate df by particle~~~~
	# """
    avg_dist = df.groupby('particle')['dist_to_boundary'].mean()
    particles = sorted(df['particle'].unique())
    for particle in particles:
        curr_df = df[ df['particle']==particle ]

        # """
    	# ~~~~add 'v_max'~~~~
    	# """
        v_max = curr_df['v'].max()
        df.loc[df['particle']==particle, 'v_max'] = v_max

        # """
    	# ~~~~add 'travel_dist'~~~~
    	# """
        travel_dist = ((curr_df['x'].max() - curr_df['x'].min())**2 + \
                    (curr_df['y'].max() - curr_df['y'].min())**2) ** 0.5
        df.loc[df['particle']==particle, 'travel_dist'] = travel_dist

        # """
    	# ~~~~add 'lifetime'~~~~
    	# """
        df.loc[df['particle']==particle, 'lifetime'] = \
                    curr_df['frame'].max() - curr_df['frame'].min() + 1

        # """
    	# ~~~~add 'boundary_type'~~~~
    	# """
        if sorters!=None and sorters['DIST_TO_BOUNDARY'] != None:
            if avg_dist[particle] >= sorters['DIST_TO_BOUNDARY'][0] \
            and avg_dist[particle] <= sorters['DIST_TO_BOUNDARY'][1]:
                df.loc[df['particle']==particle, 'boundary_type'] = 'Boundary'
            elif avg_dist[particle] < sorters['DIST_TO_BOUNDARY'][0]:
                df.loc[df['particle']==particle, 'boundary_type'] = 'Inside'
            else:
                df.loc[df['particle']==particle, 'boundary_type'] = '--none--'

            # """
        	# ~~~~add 'particle_type': Endocytosis~~~~
        	# """
            curr_df = curr_df.sort_values(by='frame', ascending=True)
            n = len(curr_df.index)
            # m = int(round(len(curr_df.index)*0.1))
            # start_depth = curr_df.head(m)['dist_to_boundary'].mean()
            # end_depth = curr_df.tail(m)['dist_to_boundary'].mean()
            # if (start_depth-end_depth)>=5 \
            # and avg_dist[particle] <= sorters['DIST_TO_BOUNDARY'][1]:
            #     df.loc[df['particle']==particle, 'particle_type'] = 'Endocytosis'
            # else:
            #     df.loc[df['particle']==particle, 'particle_type'] = '--none--'

            d_start = curr_df.iloc[0]['dist_to_boundary']
            d_end = curr_df.iloc[-1]['dist_to_boundary']
            dn = int(round(len(curr_df.index)*0.05))
            m0 = 0
            m1 = int(round(len(curr_df.index)*0.1))
            m2 = int(round(len(curr_df.index)*0.4))
            m3 = int(round(len(curr_df.index)*0.6))
            m4 = int(round(len(curr_df.index)*0.9))
            d0 = curr_df.iloc[:dn]['dist_to_boundary'].mean()
            d1 = curr_df.iloc[m1:m1+dn]['dist_to_boundary'].mean()
            d2 = curr_df.iloc[m2:m2+dn]['dist_to_boundary'].mean()
            d3 = curr_df.iloc[m3:m3+dn]['dist_to_boundary'].mean()
            d4 = curr_df.iloc[-dn:]['dist_to_boundary'].mean()
            # if d_start-d_end>=2 \
            # and d_end<sorters['DIST_TO_BOUNDARY'][0] \
            # and d_start>sorters['DIST_TO_BOUNDARY'][0]-3 \
            # and df.loc[df['particle']==particle, 'lifetime'].mean()<350:
            #     df.loc[df['particle']==particle, 'particle_type'] = 'Endocytosis'
            if d_end-d_start>=2 \
            and d_start<sorters['DIST_TO_BOUNDARY'][0] \
            and d_end>sorters['DIST_TO_BOUNDARY'][0]-3 \
            and df.loc[df['particle']==particle, 'lifetime'].mean()<350:
                df.loc[df['particle']==particle, 'particle_type'] = 'Exocytosis'
            else:
                df.loc[df['particle']==particle, 'particle_type'] = '--none--'


    return df
