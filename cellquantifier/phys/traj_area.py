from sklearn import mixture
import numpy as np

def add_traj_area(physdf):
    """
    Add columns to physdf: 'traj_sigx', 'traj_sigy', 'traj_lc', 'traj_area'

    Parameters
    ----------
    physdf : DataFrame
        DataFrame containing 'x', 'y', 'frame', 'particle'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'traj_sigx', 'traj_sigy', 'traj_lc', 'traj_area' columns
    """

    particles = physdf['particle'].unique()

    for particle in particles:
        X_train = physdf[ physdf['particle']==particle ]
        X_train = X_train.loc[:, ['x', 'y']].to_numpy()

        clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
        clf.fit(X_train)

        traj_sigx = np.sqrt(clf.covariances_[0, 0, 0])
        traj_sigy = np.sqrt(clf.covariances_[0, 1, 1])
        traj_lc = np.sqrt(clf.covariances_[0, 0, 0] + clf.covariances_[0, 1, 1])
        traj_area = np.pi * traj_sigx * traj_sigy

        physdf.loc[physdf['particle']==particle, 'traj_sigx'] = traj_sigx
        physdf.loc[physdf['particle']==particle, 'traj_sigy'] = traj_sigy
        physdf.loc[physdf['particle']==particle, 'traj_lc'] = traj_lc
        physdf.loc[physdf['particle']==particle, 'traj_area'] = traj_area

    return physdf
