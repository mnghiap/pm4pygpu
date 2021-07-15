from pm4pygpu.constants import Constants
from pm4pygpu.start_end_activities import get_end_activities
import sys

def get_variants_df(df):
	vdf = df.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY_CODE: "collect", Constants.TARGET_EV_IDX: "count"}).reset_index()
	vars_list = vdf[Constants.TARGET_ACTIVITY_CODE].to_arrow().to_pylist()
	vars_list = list(map(lambda x: Constants.VARIANTS_SEP.join([str(e) for e in x]), vars_list))
	vdf[Constants.TEMP_COLUMN_1] = vars_list
	return vdf

def filter_on_variants(df, allowed_variants):
	activities = df[Constants.TARGET_ACTIVITY].cat.categories.to_arrow().to_pylist()
	activities = {activities[i]: i for i in range(len(activities))}
	varstrings = list(map(lambda x: Constants.VARIANTS_SEP.join([str(activities[act]) for act in x.split(Constants.VARIANTS_SEP)]), allowed_variants))
	vdf = get_variants_df(df)
	vdf = vdf[vdf[Constants.TEMP_COLUMN_1].isin(varstrings)]
	return df[df[Constants.TARGET_CASE_IDX].isin(vdf[Constants.TARGET_CASE_IDX])]

def get_variants(df, max_des_vars_num=sys.maxsize, return_list=False):
	activities = df[Constants.TARGET_ACTIVITY].cat.categories.to_arrow().to_pylist()
	activities = {i: activities[i] for i in range(len(activities))}
	vdf = get_variants_df(df)
	vars_count = vdf[Constants.TEMP_COLUMN_1].value_counts()
	if max_des_vars_num < sys.maxsize:
		nlarg_vars = list(vars_count.nlargest(max_des_vars_num).to_pandas().to_dict().keys())
		vdf = vdf[vdf[Constants.TEMP_COLUMN_1].isin(nlarg_vars)]
		vars_count = vdf[Constants.TEMP_COLUMN_1].value_counts()
	res = vars_count.to_pandas().to_dict()
	res = {Constants.VARIANTS_SEP.join([activities[int(i)] for i in key.split(Constants.VARIANTS_SEP)]): res[key] for key in res.keys()}
	if return_list:
		res = [{"variant": x, "count": y} for x, y in res.items()]
	return res
