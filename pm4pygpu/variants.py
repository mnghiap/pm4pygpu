from pm4pygpu.constants import Constants
import sys
import cudf

def get_variants_df(df):
	df = df.copy()
	df[Constants.TARGET_ACTIVITY] = df[Constants.TARGET_ACTIVITY].astype("string")
	vdf = df.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY: "collect", Constants.TARGET_EV_IDX: "count"}).reset_index()
	vars_list = vdf[Constants.TARGET_ACTIVITY].to_pandas().apply(lambda x: Constants.VARIANTS_SEP.join(x))
	vdf[Constants.TARGET_ACTIVITY] = cudf.Series.from_pandas(vars_list)
	return vdf

def filter_on_variants(df, allowed_variants):
	vdf = get_variants_df(df)
	vdf = vdf[vdf[Constants.TARGET_ACTIVITY].isin(allowed_variants)]
	return df[df[Constants.TARGET_CASE_IDX].isin(vdf[Constants.TARGET_CASE_IDX])]

def get_variants(df, max_des_vars_num=sys.maxsize, return_list=False):
	vdf = get_variants_df(df)
	vars_count = vdf[Constants.TARGET_ACTIVITY].value_counts()
	if max_des_vars_num < sys.maxsize:
		vars_count = vars_count.nlargest(max_des_vars_num)
	res = vars_count.to_pandas().to_dict()
	if return_list:
		res = [{"variant": x, "count": y} for x, y in res.items()]
	return res
