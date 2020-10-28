import numpy as np
from ..qmath.ransac import ransac_polyfit

def add_ranpofit_value(
        df, indep_col, dep_col,
        cat_col=None,
        new_col_name=None,
        poly_deg=2,
        sample_ratio=0.8,
        residual_thres=0.1,
        max_trials=1000):

    """
    Categorize the df based on cat_col.
    Add a ransac_polyfit value column based on indep_col and dep_col.

    Pseudo code
    ----------
    1. Check df
    2. Sort df based on indep_col
    3. Prepare cats based on cat_col
    4. Prepare new_col_name
    5. Add ranpofit value accordingly

    Parameters
    ----------
    df : DataFrame
		DataFrame contains with cat_col, indep_col, dep_col.

    cat_col : str
        Column to use for categorical sorting.

    indep_col : str
        Columns to use as the independent value for the fitting.

    dep_col : str
        Columns to use as the dependent value for the fitting.

    new_col_name : str, optional
        Column name for the add value.

    poly_deg : int, optional
        'poly_deg' for ransac_polyfit()

    sample_ratio : float, optional
        sample_ratio to decide 'min_sample_num' for ransac_polyfit()

    residual_thres : int, optional
        'residual_thres' for ransac_polyfit()

    max_trials : int, optional
        'max_trials' for ransac_polyfit()

    Returns
    -------
    Add ransac_polyfit value column in df.

    Examples
	--------
    import pandas as pd
    import numpy as np
    from cellquantifier.phys import *

    a = np.zeros((10, 2))
    a[:,0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    a[:, 1] = [-0.2, 1.2, 3.8, 9.2, 15.8, 25.2, 35.8, 49.2, 63.8, 81.2]
    df = pd.DataFrame(a, columns=['x', 'y'])
    print(df)
    df = add_ranpofit_value(df, 'x', 'y')
    print(df)
    """

    # """
    # ~~~~~~~~Check df~~~~~~~~
    # """
    if df.empty \
    and indep_col not in df.columns \
    and dep_col not in df.columns:
    	return

    # """
    # ~~~~~~~~Sort df based on indep_col~~~~~~~~
    # """
    df.sort_values(by=indep_col)

    # """
    # ~~~~~~~~Prepare cats based on cat_col~~~~~~~~
    # """
    if cat_col:
        cats = sorted(df[cat_col].unique())
    else:
        cat_col = 'cats_helper'
        df[cat_col] = 'cat'
        cats = ['cat']

    # """
    # ~~~~~~~~Prepare new_col_name~~~~~~~~
    # """
    if new_col_name:
        new_col_name = new_col_name
    else:
        new_col_name = dep_col + '_ranpofit'

    # """
    # ~~~~~~~~Add ranpofit value accordingly~~~~~~~~
    # """
    for cat in cats:
        curr_df = df[ df[cat_col]==cat ]

        curr_indep = curr_df[indep_col].to_numpy()
        curr_dep = curr_df[dep_col].to_numpy()

        poly_param = ransac_polyfit(
                    curr_indep, curr_dep,
                    poly_deg=poly_deg,
                    min_sample_num=int(len(curr_indep)*sample_ratio),
                    residual_thres=residual_thres,
                    max_trials=max_trials,
                    )

        curr_fitting_model = np.poly1d(poly_param)
        curr_fitting_value = curr_fitting_model(curr_indep)

        df.loc[df[cat_col]==cat, new_col_name] = curr_fitting_value

    if 'cats_helper' in df.columns:
        df = df.drop(['cats_helper'], axis=1)

    return df
