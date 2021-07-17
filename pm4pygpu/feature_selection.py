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
