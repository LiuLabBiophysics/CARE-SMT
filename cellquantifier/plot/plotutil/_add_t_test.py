import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ...qmath import t_test


def add_t_test(ax, blobs_df, cat_col, hist_col,
                drop_duplicates=True,
                text_pos=[0.95, 0.3],
                color=(0,0,0,1),
                fontname='Arial',
                fontweight='normal',
                fontsize=8,
                prefix_str='',
                horizontalalignment='right',
                format='basic',
                ):

    # """
    # ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
    # """

    if blobs_df.empty:
    	return

    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """

    cats = sorted(blobs_df[cat_col].unique())
    blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]

    if len(cats) == 2:
        if drop_duplicates:
            t_stats = t_test(blobs_dfs[0].drop_duplicates('particle')[hist_col],
                            blobs_dfs[1].drop_duplicates('particle')[hist_col])
        else:
            t_stats = t_test(blobs_dfs[0][hist_col],
                            blobs_dfs[1][hist_col])
        if format=='basic':
            if t_stats[1] < .0001:
                t_test_str = prefix_str + 'P < .0001'
            elif t_stats[1] >= .0001 and t_stats[1] < .001:
                t_test_str = prefix_str + 'P < .001'
            elif t_stats[1] >= .001 and t_stats[1] < .01:
                t_test_str = prefix_str + 'P < .01'
            elif t_stats[1] >= .01 and t_stats[1] < .05:
                t_test_str = prefix_str + 'P < .05'
            else:
                t_test_str = prefix_str + 'P = %.2F' % (t_stats[1])

        if format=='general':
            if t_stats[1] >= .01:
                t_test_str = prefix_str + 'P = %.2F' % (t_stats[1])
            else:
                t_test_str = prefix_str + 'P = %.1E' % (t_stats[1])

        ax.text(text_pos[0],
                text_pos[1],
                t_test_str,
                horizontalalignment=horizontalalignment,
                verticalalignment='bottom',
                color=color,
                family=fontname,
                fontweight=fontweight,
                fontsize=fontsize,
                transform=ax.transAxes)
