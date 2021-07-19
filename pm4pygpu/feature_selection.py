from pm4pygpu.constants import Constants
from pm4pygpu.cases_df import get_first_df, get_last_df

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
	fea_df = fea_df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_num_events(df, fea_df, col_name="numEvents"):
	'''
	Select number of events for each case.
	'''
	cases_df = build_cases_df(df)[[Constants.TARGET_CASE_IDX, Constants.TARGET_EV_IDX]].rename(columns={Constants.TARGET_EV_IDX: col_name})
	fea_df = fea_df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_paths(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, column value att@v1->v2=0 if there is no path v1->v2 happens in the case, elsewhile = 1
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
