import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..phys import *
from ..util import get_frac_df
from ..plot.plotutil import *

from matplotlib.lines import Line2D
from scipy.stats import sem

def fig_quick_rna_3(merged_blobs_df, norm_row_props,
					norm_nest_col, typ_arr=['type2','type4']):

	"""

	Pseudo code
	----------
	1. Build the figure
	2. Group by 'label', 'cell_type' and 'prefix' columns

	Parameters
	----------
	input_dir : str,
		Directory containing the files of interest
	typ_arr: list
		List of unique types some of which may be in merged_xxx_df

	"""

	fig, ax = plt.subplots(1,2)

	# """
	# ~~~~~~~~~~~Copy number pivot table~~~~~~~~~~~~~~
	# """


	cols = ['cell_type', 'sample_type', 'label', 'exp_label']
	merged_blobs_df = merged_blobs_df[cols + ['region_type', 'prefix']]
	merged_blobs_df = merged_blobs_df.rename(columns={'prefix':'count_'})

	count_df_pivot = pd.pivot_table(merged_blobs_df,
									index=cols,
									columns=['region_type'],
									aggfunc='count',
									fill_value=0)


	count_df = count_df_pivot.reset_index()
	count_df.columns = [''.join(col) for col in count_df.columns]
	count_df['count_total'] = count_df['count_cyto'] + count_df['count_nuc']

	# """
	# ~~~~~~~~~~~Statistics~~~~~~~~~~~~~~
	# """

	data_col = 'count_total'; cat_cols = ['cell_type', 'exp_label']
	count_df_stat = count_df[[data_col] + cat_cols]
	count_df_stat = count_df_stat.groupby(cat_cols, sort=True)[data_col].agg([np.mean, sem])
	count_df_stat = count_df_stat.reset_index()

	count_df_stat = norm_df(count_df_stat,
					   col='mean',
					   col_arr=['mean', 'sem'],
					   row_props=norm_row_props,
					   nest_col=norm_nest_col)


	print(count_df_stat)



	bar_width = .2
	colors=['blue', 'red']
	ind_arr, mean_arr, err_arr = add_grp_bar(ax[0],
									count_df_stat,
									cat_cols=cat_cols,
									bar_width=bar_width,
									colors=colors,
									alpha=.75,
									grp_space=0.5)

	format_ax(ax[0],
			  ylabel='Relative RNA Expression',
			  ax_is_box=False,
			  xscale=[None,None,1,None],
			  show_legend=True,
			  tklabel_fontweight='bold',
			  label_fontweight='bold')

	custom_lines = [Line2D([0], [0], color=color, lw=4) \
				  for color in colors]

	ax[0].legend(custom_lines, sorted(count_df['exp_label'].unique()))

	typ_arr.insert(0, 0)
	ax[0].set_xticklabels(typ_arr)

	# """
	# ~~~~~~~~~~~Perform statistical tests~~~~~~~~~~~~~~
	# """

	ind_arr = np.delete(np.array(ind_arr), 1, 1)
	mean_arr = np.delete(np.array(mean_arr), 1, 1)
	err_arr = np.delete(np.array(err_arr), 1, 1)

	df_arr = nest_df(count_df, cat_cols=['cell_type'])
	for i, this_df in enumerate(df_arr):
		p_val = get_binary_pval(this_df, cat_col='exp_label', data_col='count')
		add_stat_anno(ax[0],
					  x_arr=ind_arr[i],
					  y_arr=np.array(mean_arr[i])+np.array(err_arr[i]),
					  p_val_arr=p_val,
					  bwidth=.5*bar_width,
					  delta=0,
					  show_brackets=False)

	# """
	# ~~~~~~~~~~~Nucleus/Cytoplasm Fractions~~~~~~~~~~~~~~
	# """

	label_map = merged_blobs_df[['label', 'exp_label']].drop_duplicates('label')

	frac_df = get_frac_df(merged_blobs_df,
						  frac_col='label',
						  group_cols=['label','cell_type','region_type'],
						  map_df=label_map,
						  map_col='exp_label')

	# frac_df.to_csv('/home/clayton/Desktop/data-analysis/200521_HLA-DMB-final-analysis/200521_HLA-DMB-fig3-frac-df.csv')

	stack_col = 'region_type'
	data_col = 'fracs'; cat_cols = ['cell_type', 'exp_label']
	frac_df_stat = frac_df[[data_col] + cat_cols + [stack_col]]
	frac_df_stat = frac_df_stat.groupby(cat_cols+[stack_col], \
								sort=True)[data_col].agg([np.mean, sem])
	frac_df_stat = frac_df_stat.reset_index()

	# frac_df_stat.to_csv('/home/clayton/Desktop/data-analysis/200521_HLA-DMB-final-analysis/200521_HLA-DMB-fig3-frac-df-stat.csv')

	ind_arr, mean_arr, err_arr = add_grp_bar(ax[1],
										    frac_df_stat,
											cat_cols=cat_cols,
											stack_col=stack_col,
											colors=['blue', 'red'],
											bar_width=.2,
											alpha=.75,
											grp_space=.5,
											btm_clr=['black', 'black'])

	format_ax(ax[1],
			  ylabel='Fraction of total RNA',
			  ax_is_box=False,
			  xscale=[None,None,1,None],
			  show_legend=True,
			  tklabel_fontweight='bold',
			  label_fontweight='bold')

	custom_lines = [Line2D([0], [0], color='red', lw=4),
					Line2D([0], [0], color='blue', lw=4)]




	# """
	# ~~~~~~~~~~~Perform statistical tests~~~~~~~~~~~~~~
	# """

	df_arr = nest_df(frac_df, cat_cols=['cell_type', 'region_type'])
	p_val_arr = df_to_pval(df_arr, cat_col='exp_label', data_col='fracs')

	add_stat_anno(ax[1],
				  x_arr=[0.5, 0.5, 1.0, 1.0],
				  y_arr=[.625, 1.05, .625, 1.05],
				  p_val_arr=p_val_arr,
				  bwidth=.5*bar_width,
				  bheight=0.01,
				  delta=0,
				  show_brackets=True)

	custom_lines = [Line2D([0], [0], color='blue', lw=4),
					Line2D([0], [0], color='black', alpha=0.3, lw=4)]


	ax[1].legend(custom_lines, ['nucleus', 'cytoplasm'])


	plt.tight_layout()
