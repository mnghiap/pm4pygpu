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
	For an attribute att and two values v1, v2, column value att@v1->v2=0 if no such directly-follow happens in the case, elsewhile = #occurences of the path.
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
	#df = df.query(Constants.TARGET_CASE_IDX + "==" + Constants.TARGET_PRE_CASE).groupby([Constants.TARGET_CASE_IDX, Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2]).count().reset_index()
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
