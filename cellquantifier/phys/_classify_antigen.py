import pandas as pd
import numpy as np
import trackpy as tp
from ..smt.track import get_d_values

def classify_antigen(df):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['', ]:
        if col in df:
            df = df.drop(col, axis=1)

    # """
	# ~~~~Iterate df by particle~~~~
	# """
    particles = sorted(df['particle'].unique())

    ind = 1
    tot = len(particles)
    for particle in particles:
        print("Classify antigen (%d/%d)" % (ind, tot))
        ind = ind + 1

        curr_df = df[ df['particle']==particle ]
        curr_df = curr_df.sort_values(['particle', 'frame'])

        # """
    	# ~~~~Classify subparticle type: Directional Motion (DM)~~~~
    	# """
        # footprints that meet DM requirements
        DM = curr_df['local_alpha'] >= 1.2

        # DM subparticle label
        DM_ptcl = (DM != DM.shift()).cumsum() - 1
        DM_ptcl = curr_df['particle'].apply(str) + '_DM_' + DM_ptcl.apply(str)
        DM_ptcl[ DM==False ] = ''

        # DM subparticle_type label
        DM_ptcl_type = DM.copy()
        DM_ptcl_type[ DM ] = 'DM'
        DM_ptcl_type[ DM==False ] = ''


        # """
    	# ~~~~Classify subparticle type: Brownion Motion (BM)~~~~
    	# """
        # footprints that meet BM requirements
        BM = (curr_df['local_alpha'] > 0.8) & (curr_df['local_alpha'] < 1.2)

        # BM subparticle label
        BM_ptcl = (BM != BM.shift()).cumsum() - 1
        BM_ptcl = curr_df['particle'].apply(str) + '_BM_' + BM_ptcl.apply(str)
        BM_ptcl[ BM==False ] = ''

        # BM subparticle_type label
        BM_ptcl_type = BM.copy()
        BM_ptcl_type[ BM ] = 'BM'
        BM_ptcl_type[ BM==False ] = ''


        # """
    	# ~~~~Classify subparticle type: Confined Motion (CM)~~~~
    	# """
        # footprints that meet CM requirements
        CM = (curr_df['local_alpha'] <= 0.8) & (curr_df['local_alpha'] > 0.2)

        # CM subparticle label
        CM_ptcl = (CM != CM.shift()).cumsum() - 1
        CM_ptcl = curr_df['particle'].apply(str) + '_CM_' + CM_ptcl.apply(str)
        CM_ptcl[ CM==False ] = ''

        # CM subparticle_type label
        CM_ptcl_type = CM.copy()
        CM_ptcl_type[ CM ] = 'CM'
        CM_ptcl_type[ CM==False ] = ''


        # """
    	# ~~~~Combine and generate 'subparticle', 'subparticle_type' col~~~~
    	# """
        df.loc[df['particle']==particle, 'subparticle'] = \
                DM_ptcl + BM_ptcl + CM_ptcl
        df.loc[df['particle']==particle, 'subparticle_type'] = \
                DM_ptcl_type + BM_ptcl_type + CM_ptcl_type


    # """
	# ~~~~Calculte properties for subparticle~~~~
	# """
    sub_ptcls = sorted(df['subparticle'].unique())

    ind = 1
    tot = len(sub_ptcls)

    for sub_ptcl in sub_ptcls:
        print("Add more subparticle property (%d/%d)" % (ind, tot))
        ind = ind + 1

        curr_df = df[ df['subparticle']==sub_ptcl ]

        sp_D = curr_df['local_D'].mean()
        df.loc[df['subparticle']==sub_ptcl, 'subparticle_D'] = sp_D

        sp_alpha = curr_df['local_alpha'].mean()
        df.loc[df['subparticle']==sub_ptcl, 'subparticle_alpha'] = sp_alpha

        sp_dir_pers = curr_df['dir_pers'].mean()
        df.loc[df['subparticle']==sub_ptcl, 'subparticle_dir_pers'] = sp_dir_pers

        sp_traj_length = len(curr_df)
        df.loc[df['subparticle']==sub_ptcl, 'subparticle_traj_length'] = sp_traj_length

        sp_travel_dist = ((curr_df['x'].max() - curr_df['x'].min())**2 + \
            (curr_df['y'].max() - curr_df['y'].min())**2) ** 0.5
        df.loc[df['subparticle']==sub_ptcl, 'subparticle_travel_dist'] = sp_travel_dist

    # """
    # ~~~~Eliminate data for subparticle '' (unclassified subparticle)~~~~
    # """
    df.loc[df['subparticle']=='', 'subparticle_D'] = ''
    df.loc[df['subparticle']=='', 'subparticle_alpha'] = ''
    df.loc[df['subparticle']=='', 'subparticle_dir_pers'] = ''
    df.loc[df['subparticle']=='', 'subparticle_traj_length'] = ''
    df.loc[df['subparticle']=='', 'subparticle_travel_dist'] = ''


    return df
