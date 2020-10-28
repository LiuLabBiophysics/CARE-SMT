from cellquantifier.segm import get_dist2boundary_mask, get_dist2boundary_mask_batch

def add_dist_to_53bp1(df, thres_mask):

    """
    Label particles in a DataFrame based on dist2boundary_mask

    Parameters
    ----------
    thres_mask : ndarray
        Binary thres_mask of 53bp1
    df : DataFrame
        DataFrame containing x,y columns

    Returns
    -------
    df: DataFrame
        DataFrame with added 'dist_to_boundary' column

    Examples
    --------
    import pandas as pd; import pims
    from cellquantifier.segm import get_thres_mask
    from cellquantifier.phys import add_dist_to_53bp1

    frames = pims.open('cellquantifier/data/simulated_cell.tif')
    df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    thres_mask = get_thres_mask(frames[0], sig=1, thres_rel=0.5)
    df = add_dist_to_53bp1(df, thres_mask)
    print(df)
    """

    dist2boundary_mask = get_dist2boundary_mask(thres_mask)

    for index in df.index:
        r = int(round(df.at[index, 'x']))
        c = int(round(df.at[index, 'y']))
        df.at[index, 'dist_to_53bp1'] = dist2boundary_mask[r, c]

    return df


def add_dist_to_53bp1_batch(df, thres_masks):

    """
    Label particles in a DataFrame based on dist2boundary_masks

    Parameters
    ----------
    thres_masks : 3D ndarray
        Binary thres_masks of cell video
    df : DataFrame
        DataFrame containing 'x', 'y', 'frame' columns

    Returns
    -------
    df: DataFrame
        DataFrame with added 'dist_to_boundary' column

    Examples
    --------
    import pandas as pd
    from skimage.io import imread
    from cellquantifier.io import imshow
    from cellquantifier.segm import get_thres_mask_batch
    from cellquantifier.phys import add_dist_to_53bp1_batch

    tif = imread('cellquantifier/data/simulated_cell.tif')[0:2]
    thres_masks = get_thres_mask_batch(tif, sig=1, thres_rel=0.5)
    df = pd.read_csv('cellquantifier/data/simulated_cell-fittData.csv')
    df = add_dist_to_53bp1_batch(df, thres_masks)
    print(df[df['frame']==0])
    """

    dist2boundary_masks = get_dist2boundary_mask_batch(thres_masks)

    for i in range(len(dist2boundary_masks)):
        curr_dist2boundary_mask = dist2boundary_masks[i]
        curr_df = df[ df['frame'] == i ]

        for index in curr_df.index:
            r = int(round(df.at[index, 'x']))
            c = int(round(df.at[index, 'y']))
            df.at[index, 'dist_to_53bp1'] = curr_dist2boundary_mask[r, c]

    return df
