from pm4pygpu.constants import Constants
from pm4pygpu.cases_df import get_first_df, get_last_df, build_cases_df
from pm4pygpu.dfg import paths_udf
import numpy as np

def select_number_column(df, fea_df, col):
	df = get_last_df(df.dropna(subset=[col]))[[Constants.TARGET_CASE_IDX, col]]
	fea_df = fea_df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_string_column(df, fea_df, col):
	vals = df[col].unique().to_arrow().to_pylist()
	for val in vals:
		if val is not None:
			filt_df_cases = df[df[col].isin([val])][Constants.TARGET_CASE_IDX].unique()
			new_col = col + "_" + val.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
			fea_df[new_col] = fea_df[Constants.TARGET_CASE_IDX].isin(filt_df_cases)
			fea_df[new_col] = fea_df[new_col].astype("int")
	return fea_df

def get_features_df(df, list_columns):
	fea_df = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	for col in list_columns:
		if "object" in str(df[col].dtype):
			fea_df = select_string_column(df, fea_df, col)
		elif "float" in str(df[col].dtype) or "int" in str(df[col].dtype):
			fea_df = select_number_column(df, fea_df, col)
	fea_df = fea_df.sort_values(Constants.TARGET_CASE_IDX)
	return fea_df

def select_features(df, low_b_str=5, up_b_str=50):
	list_columns = []
	df_cases = df[Constants.TARGET_CASE_IDX].nunique()
	for col in df.columns:
		if not col.startswith("custom_") and not col.startswith("index"):
			if "object" in str(df[col].dtype):
				nuniq = df[col].nunique()
				if low_b_str <= nuniq <= up_b_str:
					list_columns.append(col)
			elif "float" in str(df[col].dtype) or "int" in str(df[col].dtype):
				filt_df_cases = df.dropna(subset=[col])[Constants.TARGET_CASE_IDX].nunique()
				if df_cases == filt_df_cases:
					list_columns.append(col)
	return list_columns

def get_automatic_features_df(df, low_b_str=5, up_b_str=50):
	list_columns = select_features(df, low_b_str=low_b_str, up_b_str=up_b_str)
	return get_features_df(df, list_columns)

def select_case_duration(df, fea_df, col_name="caseDuration"):
	'''
	Select case duration for each case.
	'''
	cases_df = build_cases_df(df)[[Constants.TARGET_CASE_IDX, Constants.CASE_DURATION]].rename(columns={Constants.CASE_DURATION: col_name})
	fea_df = fea_df.merge(cases_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_num_events(df, fea_df, col_name="numEvents"):
	'''
	Select number of events for each case.
	'''
	cases_df = build_cases_df(df)[[Constants.TARGET_CASE_IDX, Constants.TARGET_EV_IDX]].rename(columns={Constants.TARGET_EV_IDX: col_name})
	fea_df = fea_df.merge(cases_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_directly_follows_paths(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, column value att@v1->v2=0 if no such directly-follow happens in the case, elsewhile = #occurences of directly-follows.
	Assumption: df is sorted by case and timestamp as in format.py
	'''
	df = df.copy()
	df = df.sort_values(by = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP])
	vals = df[att].unique().to_arrow().to_pylist()
	att_numeric = att + '_numeric'
	df[att_numeric] = df[att].astype('category').cat.codes
	keys = df[att_numeric].to_arrow().to_pylist()
	values = df[att].to_arrow().to_pylist()
	att_dict = {key: value for key, value in zip(keys, values)}
	df[Constants.TEMP_COLUMN_2] = df[att_numeric]
	df = df.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols = [Constants.TEMP_COLUMN_2], outcols= {Constants.TEMP_COLUMN_1: np.int32})
	df = df.query(Constants.TARGET_CASE_IDX+" == "+Constants.TARGET_PRE_CASE)
	for v1 in att_dict.keys():
		for v2 in att_dict.keys():
			ev_idxs = df.query(Constants.TEMP_COLUMN_1+"=="+str(v1)+" and "+Constants.TEMP_COLUMN_2+"=="+str(v2))[Constants.TARGET_EV_IDX].unique()
			str_v1 = att_dict[v1].encode('ascii',errors='ignore').decode('ascii').replace(" ","")
			str_v2 = att_dict[v2].encode('ascii',errors='ignore').decode('ascii').replace(" ","")
			df[att+"@"+str_v1+"->"+str_v2] = df[Constants.TARGET_EV_IDX].isin(ev_idxs).astype("int")
	df = df[[Constants.TARGET_CASE_IDX] + [att+"@"+v1+"->"+v2 for v1 in att_dict.values() for v2 in att_dict.values()]]
	df = df.groupby(Constants.TARGET_CASE_IDX).sum().reset_index()
	fea_df = fea_df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_paths(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, column value att@v1->v2=0 if there is no such eventually-follows path happens in the case, elsewhile = 1
	Assumption: df is sorted by case and timestamp as in format.py
	'''
	case_df = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	df = df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	df = df.query(Constants.TARGET_TIMESTAMP + "<" + Constants.TARGET_TIMESTAMP + "_y")
	vals = df[att].unique().to_arrow().to_pylist()
	for v1 in vals:
		for v2 in vals:
			if v1 is not None and v2 is not None:
				#case_idxs = df.query(att+"=="+str(v1)+" and "+att+"_y=="+str(v2)+" and "+Constants.TARGET_TIMESTAMP+"<"+Constants.TARGET_TIMESTAMP+"_y")[Constants.TARGET_EV_IDX].unique()
				tdf = df[df[att] == v1]
				tdf = tdf[tdf[att+'_y'] == v2]
				case_idxs = tdf[Constants.TARGET_CASE_IDX].unique()
				str_v1 = v1.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				str_v2 = v2.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				case_df[att+"@"+str_v1+"->"+str_v2] = case_df[Constants.TARGET_CASE_IDX].isin(case_idxs).astype("int")
	fea_df = fea_df.merge(case_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_path_durations(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, compute duration from first occurence of v1 to last occurence of v2 in the case
	-1 if does not happen
	'''
	case_df = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	df = df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	df = df.query(Constants.TARGET_TIMESTAMP + "<" + Constants.TARGET_TIMESTAMP + "_y")
	vals = df[att].unique().to_arrow().to_pylist()
	for v1 in vals:
		for v2 in vals:
			if v1 is not None and v2 is not None:
				tdf = df[df[att] == v1]
				tdf = tdf[tdf[att+'_y'] == v2]
				tdf = tdf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_TIMESTAMP: "min", Constants.TARGET_TIMESTAMP+"_y": "max"}).reset_index()
				str_v1 = v1.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				str_v2 = v2.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				tdf["duration"+"@"+att+"@"+str_v1+"->"+str_v2] = tdf[Constants.TARGET_TIMESTAMP+"_y"] - tdf[Constants.TARGET_TIMESTAMP]
				tdf = tdf.drop([Constants.TARGET_TIMESTAMP, Constants.TARGET_TIMESTAMP+"_y"], axis=1)
				case_df = case_df.merge(tdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
				case_df["duration"+"@"+att+"@"+str_v1+"->"+str_v2] = case_df["duration"+"@"+att+"@"+str_v1+"->"+str_v2].astype("int").fillna(-1)
	fea_df = fea_df.merge(case_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_time_from_start_of_case(df, fea_df, att):
	'''
	Given a value v of attribute att, compute the time from the start of case to first/last occurence of v.
	If v does not happen in the case, the value is set to -1.
	'''
	cases_df = build_cases_df(df)
	cdf = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	vals = df[att].unique().to_arrow().to_pylist()
	for val in vals:
		if val is not None:
			val_df = df[df[att].isin([val])].groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_TIMESTAMP: ["min", "max"]}).reset_index()
			val_df.columns = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP+"_min", Constants.TARGET_TIMESTAMP+"_max"]
			val_df = val_df.merge(cases_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
			str_val = val.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
			val_df["timeToFirst@"+att+"@"+str_val] = val_df[Constants.TARGET_TIMESTAMP+"_min"] - val_df[Constants.TARGET_TIMESTAMP]
			val_df["timeToLast@"+att+"@"+str_val] = val_df[Constants.TARGET_TIMESTAMP+"_max"] - val_df[Constants.TARGET_TIMESTAMP]
			val_df = val_df[[Constants.TARGET_CASE_IDX, "timeToFirst@"+att+"@"+str_val, "timeToLast@"+att+"@"+str_val]]
			cdf = cdf.merge(val_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
			cdf["timeToFirst@"+att+"@"+str_val] = cdf["timeToFirst@"+att+"@"+str_val].astype("int").fillna(-1)
			cdf["timeToLast@"+att+"@"+str_val] = cdf["timeToLast@"+att+"@"+str_val].astype("int").fillna(-1)
	fea_df = fea_df.merge(cdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df


def select_time_to_end_of_case(df, fea_df, att):
	'''
	Given a value v of attribute att, compute the time to the end of case from first/last occurence of v.
	If v does not happen in the case, the value is set to -1.
	'''
	cases_df = build_cases_df(df)
	cdf = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	vals = df[att].unique().to_arrow().to_pylist()
	for val in vals:
		if val is not None:
			val_df = df[df[att].isin([val])].groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_TIMESTAMP: ["min", "max"]}).reset_index()
			val_df.columns = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP+"_min", Constants.TARGET_TIMESTAMP+"_max"]
			val_df = val_df.merge(cases_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
			str_val = val.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
			val_df["timeFromFirst@"+att+"@"+str_val] = val_df[Constants.TARGET_TIMESTAMP+"_2"] - val_df[Constants.TARGET_TIMESTAMP+"_min"]
			val_df["timeFromLast@"+att+"@"+str_val] = val_df[Constants.TARGET_TIMESTAMP+"_2"] - val_df[Constants.TARGET_TIMESTAMP+"_max"]
			val_df = val_df[[Constants.TARGET_CASE_IDX, "timeFromFirst@"+att+"@"+str_val, "timeFromLast@"+att+"@"+str_val]]
			cdf = cdf.merge(val_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
			cdf["timeFromFirst@"+att+"@"+str_val] = cdf["timeFromFirst@"+att+"@"+str_val].astype("int").fillna(-1)
			cdf["timeFromLast@"+att+"@"+str_val] = cdf["timeFromLast@"+att+"@"+str_val].astype("int").fillna(-1)
	fea_df = fea_df.merge(cdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_combinations(df, fea_df, att1, att2):
	'''
	Select value combinations of two atrtibutes att1, att2, e.g. att1@att2=v1@v2
	'''
	cdf = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	vals1 = df[att1].unique().to_arrow().to_pylist()
	vals2 = df[att2].unique().to_arrow().to_pylist()
	for v1 in vals1:
		for v2 in vals2:
			if v1 is not None and v2 is not None:
				val_df = df[df[att1].isin([v1])]
				val_df = val_df[val_df[att2].isin([v2])]
				case_idxs = val_df[Constants.TARGET_CASE_IDX].unique()
				str_v1 = v1.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				str_v2 = v2.encode('ascii',errors='ignore').decode('ascii').replace(" ","")
				cdf[att1+"@"+att2+"="+str_v1+"@"+str_v2] = cdf[Constants.TARGET_CASE_IDX].isin(case_idxs).astype("int")
	fea_df = fea_df.merge(cdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_num_cases_in_progress(df, fea_df, col_name="casesInProgress"):
	'''
	Select number of cases opened at the same time for each case
	'''
	cdf = df[Constants.TARGET_CASE_IDX].unique().to_frame()
	cases_df = build_cases_df(df)
	cases_df[Constants.TEMP_COLUMN_1] = 0
	cases_df = cases_df.merge(cases_df, on=[Constants.TEMP_COLUMN_1], how="left", suffixes=('','_y'))
	cases_df = cases_df.query(Constants.TARGET_TIMESTAMP+"<"+Constants.TARGET_TIMESTAMP+"_2_y and "+Constants.TARGET_TIMESTAMP+"_y <"+Constants.TARGET_TIMESTAMP+"_2")
	cases_df = cases_df.groupby(Constants.TARGET_CASE_IDX).count().reset_index()[[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP]]
	cdf = cdf.merge(cases_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	cdf[Constants.TARGET_TIMESTAMP] = cdf[Constants.TARGET_TIMESTAMP].astype("int").fillna(0)
	cdf = cdf.rename(columns={Constants.TARGET_TIMESTAMP: col_name})
	fea_df = fea_df.merge(cdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_resource_workload_during_case(df, fea_df):
	'''
	Select workload of all resources OF A CASE during timeframe of that case, i.e. number of events performed from start to end in the case
	Value is 0 if resource does not belong to the case.
	'''
	cases_df = build_cases_df(df)
	cases = cases_df[Constants.TARGET_CASE_IDX].to_arrow().to_pylist()
	start_time = cases_df[Constants.TARGET_TIMESTAMP].to_arrow().to_pylist()
	start_time = {cases[i]: start_time[i] for i in range(len(start_time))}
	end_time = cases_df[Constants.TARGET_TIMESTAMP+"_2"].to_arrow().to_pylist()
	end_time = {cases[i]: end_time[i] for i in range(len(end_time))}
	rdf = df[Constants.TARGET_RESOURCE].unique().to_frame()
	for case in cases:
		rsrc_in_case = df[df[Constants.TARGET_CASE_IDX] == case][Constants.TARGET_RESOURCE].unique().to_arrow().to_pylist()
		cdf = df[df[Constants.TARGET_RESOURCE].astype("string").isin(rsrc_in_case)] 
		cdf = df.query(Constants.TARGET_TIMESTAMP+">="+str(start_time[case])+" and "+Constants.TARGET_TIMESTAMP+"<="+str(end_time[case]))
		cdf = cdf.groupby([Constants.TARGET_RESOURCE]).agg({Constants.TARGET_EV_IDX: "count"}).reset_index()
		rdf = rdf.merge(cdf, on=[Constants.TARGET_RESOURCE], how='left', suffixes=('','_y'))
		rdf = rdf.rename(columns = {Constants.TARGET_EV_IDX: case})
	rdf = rdf.fillna(0)
	rdf.index = rdf[Constants.TARGET_RESOURCE]
	rdf = rdf.drop([Constants.TARGET_RESOURCE], axis=1)
	rdf = rdf.T.reset_index().rename(columns={'index':Constants.TARGET_CASE_IDX})
	fea_df = fea_df.merge(rdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df
