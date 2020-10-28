import matplotlib.pyplot as plt
import seaborn as sns

def add_violin_1(ax, df, data_col, cat_col):
    """
    Add violin plot in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
		DataFrame containing data_col, cat_col

    data_col : str
		Column contain the data

    cat_col : str
		Column to use for categorical sorting

    Returns
    -------
    Add violin plots in the ax.
    """

    cats = sorted(df[cat_col].unique())
    sns.set(style="white", palette="muted", color_codes=True)

    # """
    # ~~~~~~~~~~~Check~~~~~~~~~~~~~~
    # """

    if df.empty or len(cats)!=2:
        print("##############################################")
        print("ERROR: df is empty, or len(cats) is not 2 !!!")
        print("##############################################")
        return

    # """
    # ~~~~~~~~~~~plot violin~~~~~~~~~~~~~~
    # """

    sns.violinplot(x=[cats[0] + ' vs ' + cats[1]]*len(df),
                y=data_col, hue=cat_col,
                data=df, split=True, inner="quartile", ax=ax)

    ax.legend().set_title('')
    ax.legend(loc='upper left', frameon=False, fontsize=15)



def add_violin_2(ax, df, data_col, cat_col,
                hue_order=None,
                inner='quartile',
                RGBA_alpha=0.6,
                linewidth=0.5,
                set_format=True):
    """
    Add violin plot in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
		DataFrame containing data_col, cat_col

    data_col : str
		Column contain the data

    cat_col : str
		Column to use for categorical sorting

    Returns
    -------
    Add violin plots in the ax.
    """

    cats = sorted(df[cat_col].unique())
    sns.set(style="white", palette="coolwarm", color_codes=True)

    # """
    # ~~~~~~~~~~~Check~~~~~~~~~~~~~~
    # """

    if df.empty or len(cats)!=2:
        print("##############################################")
        print("ERROR: df is empty, or len(cats) is not 2 !!!")
        print("##############################################")
        return

    # """
    # ~~~~~~~~~~~plot split violin use seaborn~~~~~~~~~~~~~~
    # """
    sns.violinplot(x=[hue_order[0] + ' vs ' + hue_order[1]]*len(df),
                y=data_col,
                data=df,
                split=True,
                hue=cat_col,
                hue_order=hue_order,
                inner=inner,
                ax=ax)

    # """
    # ~~~~~~~~~~~update violin colors with coolwarm~~~~~~~~~~~~~~
    # """
    # Define colors
    warm = plt.cm.coolwarm(0.99)
    cool = plt.cm.coolwarm(0)
    warm_alpha = (warm[0], warm[1], warm[2], RGBA_alpha)
    cool_alpha = (cool[0], cool[1], cool[2], RGBA_alpha)

    # update violin colors
    violin_parts = ax.collections
    plt.setp(violin_parts[0],
            facecolor=cool_alpha,
            edgecolor=cool,
            linewidths=linewidth,
            )
    plt.setp(violin_parts[1],
            facecolor=warm_alpha,
            edgecolor=warm,
            linewidths=linewidth,
            )

    # update legend colors
    handles, labels = ax.get_legend_handles_labels()
    plt.setp(handles[0],
            facecolor=cool_alpha,
            edgecolor=cool)
    plt.setp(handles[1],
            facecolor=warm_alpha,
            edgecolor=warm)
    ax.legend(handles, labels)

    # update lines colors
    lines = ax.get_lines()
    for i in range(3):
        plt.setp(lines[i],
            color=cool,
            linewidth=linewidth)
    for i in range(3,6):
        plt.setp(lines[i],
            color=warm,
            linewidth=linewidth)

    """
    ~~~~~~~~~~~Set ax format~~~~~~~~~~~~~~
    """
    if set_format:
        ax.legend(loc='upper left', frameon=False, fontsize=8)
