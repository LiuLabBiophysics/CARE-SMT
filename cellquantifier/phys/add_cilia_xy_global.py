import numpy as np
import math
from ..qmath import *
from .add_ranpofit_value import *

def add_cilia_x_global_raw(df):
    """
    Add 'x_global_raw' column in df using add_ranpofit_value().

    Parameters
    ----------
    df : DataFrame
		DataFrame contains with 'frame', 'x', 'particle'.

    Returns
    -------
    Add 'x_global' column in df.

    Examples
	--------
    import pandas as pd
    import numpy as np
    from cellquantifier.phys import *
    import matplotlib.pyplot as plt

    a = np.zeros((10, 3))
    a[:,0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a[:, 1] = [-0.2, 1.2, 3, 10, 14, 27, 33, 52, 60, 85]
    a[:, 2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame(a, columns=['frame', 'x', 'particle'])
    print(df)
    df = add_cilia_x_global_raw(df)
    print(df)
    fig, ax = plt.subplots()
    ax.plot(df['frame'], df['x'], 'ro')
    ax.plot(df['frame'], df['x_global'])
    plt.show()
    """
    df = add_ranpofit_value(df, 'frame', 'x',
            cat_col='particle',
            new_col_name='x_global',
            poly_deg=2,
            sample_ratio=0.8,
            residual_thres=0.1,
            max_trials=1000)
    return df


def add_cilia_y_global_raw(df):
    """
    Add 'y_global_raw' column in df using add_ranpofit_value().

    Parameters
    ----------
    df : DataFrame
		DataFrame contains with 'frame', 'y', 'particle'.

    Returns
    -------
    Add 'y_global' column in df.

    Examples
	--------
    import pandas as pd
    import numpy as np
    from cellquantifier.phys import *
    import matplotlib.pyplot as plt

    a = np.zeros((10, 3))
    a[:,0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a[:, 1] = [-0.2, 1.2, 3, 10, 14, 27, 33, 52, 60, 85]
    a[:, 2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    df = pd.DataFrame(a, columns=['frame', 'y', 'particle'])
    print(df)
    df = add_cilia_y_global_raw(df)
    print(df)
    fig, ax = plt.subplots()
    ax.plot(df['frame'], df['y'], 'ro')
    ax.plot(df['frame'], df['y_global'])
    plt.show()
    """
    df = add_ranpofit_value(df, 'frame', 'y',
            cat_col='particle',
            new_col_name='y_global',
            poly_deg=2,
            sample_ratio=0.8,
            residual_thres=0.1,
            max_trials=1000)

    return df


def add_cilia_xy_global_fine(df, manual_mode=False, angle=45):
    """
    Add columns of 'x_global', 'y_global' in df using add_ranpofit_value().
    Then replace the 'x_global', 'y_global' with the nearest point value.

    Parameters
    ----------
    df : DataFrame
		DataFrame contains with 'frame', 'y', 'particle'.

    Returns
    -------
    Add 'x_global', 'y_global' column in df.

    Examples
	--------
    import pandas as pd
    import numpy as np
    from cellquantifier.phys import *

    a = np.zeros((10, 5))
    a[:,0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a[:,1] = [-0.2, 1.2, 1.8, 3.2, 3.8, 5.2, 5.8, 7.2, 7.8, 9.2]
    a[:, 2] = [-0.2, 1.2, 3, 10, 14, 27, 33, 52, 60, 85]
    a[:, 3] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    a[:, 4] = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    df = pd.DataFrame(a, columns=['frame', 'x', 'y', 'particle', 'D'])
    print(df)
    df = add_cilia_xy_global_fine(df)
    print(df)
    """
    if manual_mode:
        x_avg = df['x'].mean()
        y_avg = df['y'].mean()
        slope = math.tan(angle/180*np.pi)
        df['x_global'] = df['x']
        df['y_global'] = slope * df['x_global'] + (y_avg - slope*x_avg)
    else:
        df = add_cilia_x_global_raw(df)
        df = add_cilia_y_global_raw(df)

    particles = sorted(df['particle'].unique())

    for particle in particles:
        curr_df = df[ df['particle']==particle ]

        frame_raw = curr_df['frame'].to_numpy()
        x_global_raw = curr_df['x_global'].to_numpy()
        y_global_raw = curr_df['y_global'].to_numpy()

        frame_fine = np.linspace(frame_raw.min()-int(len(frame_raw)*0.5),
                        frame_raw.max()+int(len(frame_raw)*0.5),
                        num=len(frame_raw)*100)
        x_global_fine = np.interp(frame_fine, frame_raw, x_global_raw)
        y_global_fine = np.interp(frame_fine, frame_raw, y_global_raw)

        X_raw = curr_df[ ['x', 'y'] ].to_numpy()
        X_global_fine = np.zeros((len(x_global_fine), 2))
        X_global_fine[:, 0] = x_global_fine
        X_global_fine[:, 1] = y_global_fine
        result = np.zeros_like(X_raw)
        for i in range(len(X_raw)):
            result[i] = get_nearest_point(X_raw[i], X_global_fine)

        df.loc[df['particle']==particle, ['x_global', 'y_global']] = result

    return df
