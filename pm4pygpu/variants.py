from pm4pygpu.constants import Constants
from pm4pygpu.start_end_activities import get_end_activities
import sys

def get_variants_df(df):
	vdf = df.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY_CODE: "collect", Constants.TARGET_EV_IDX: "count"}).reset_index()
	return vdf

def filter_on_variants(df, allowed_variants):
	activities = df[Constants.TARGET_ACTIVITY].cat.categories.to_arrow().to_pylist()
	activities = {activities[i]: i for i in range(len(activities))}
	list_tup_vars = []
	for varstri in allowed_variants:
		var = varstri.split(Constants.VARIANTS_SEP)
		v1 = 0
		v2 = 0
		for i in range(len(var)):
			v1 += activities[var[i]]
			v2 += (len(var) + i + 1) * (activities[var[i]] + 1)
		list_tup_vars.append("(" + Constants.TARGET_ACTIVITY_CODE + " == "+str(v1)+" and " +Constants.TARGET_VARIANT_NUMBER+" == "+str(v2) + ")")
	this_query = " or ".join(list_tup_vars)
	cdf = get_variants_df(df)
	cdf = cdf.query(this_query)[Constants.TARGET_CASE_IDX]
	return df[df[Constants.TARGET_CASE_IDX].isin(cdf)]

def get_variants(df, max_des_vars_num=sys.maxsize, return_list=False):
	activities = df[Constants.TARGET_ACTIVITY].cat.categories.to_arrow().to_pylist()
	activities = {i: activities[i] for i in range(len(activities))}
	vdf = get_variants_df(df)
	vars_list = vdf[Constants.TARGET_ACTIVITY_CODE].to_arrow().to_pylist()
	vars_list = list(map(lambda x: ",".join([str(e) for e in x]), vars_list))
	vdf[Constants.TEMP_COLUMN_1] = vars_list
	vars_count = vdf[Constants.TEMP_COLUMN_1].value_counts()
	if max_des_vars_num < sys.maxsize:
		nlarg_vars = list(vars_count.nlargest(max_des_vars_num).to_pandas().to_dict().keys())
		vdf = vdf[vdf[Constants.TEMP_COLUMN_1].isin(nlarg_vars)]
		vars_count = vdf[Constants.TEMP_COLUMN_1].value_counts()
	res = vars_count.to_pandas().to_dict()
	res = {",".join([activities[int(i)] for i in key.split(",")]): res[key] for key in res.keys()}
	if return_list:
		res = [{"variant": x, "count": y} for x, y in res.items()]
	return res