from pm4pygpu.constants import Constants
from pm4pygpu.cases_df import get_first_df, get_last_df, build_cases_df
from pm4pygpu.dfg import paths_udf
import numpy as np
import cudf

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
				df_col = df[[Constants.TARGET_CASE_IDX, col]].dropna(subset=[col])
				if df_col[Constants.TARGET_CASE_IDX].nunique() == df_cases:
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

@cuda.jit
def combinations_kernel(unique_combi_matrix, df_matrix, combi_case_matrix):
	'''
	This kernel computes whether a two-combination (attribute1 x attribute2) appears in a case
	unique_combi_matrix (shape: num_combinations x 2): contains all unique 2-combinations in the log
	df_matrix: (shape: num-events x 3): the df in numeric (category codes) forms. column 0 is attribute1, col 1 is attribute2, col 2 is case idx
	combi_case_matrix (shape: num_cases x num_unique_combinations)
	'''
	i = cuda.grid(1)
	if i < df_matrix.shape[0]:
		for j in range(unique_combi_matrix.shape[0]):
			if unique_combi_matrix[j][0] == df_matrix[i][0] and unique_combi_matrix[j][1] == df_matrix[i][1]:
				combi_case_matrix[df_matrix[i][2]][j] = 1

def select_attribute_directly_follows_paths(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, column value att_directly@v1->v2=0 if no such direct path happens in the case, elsewhile = 1
	'''
	df = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP, att, Constants.TARGET_PRE_CASE]]
	df = df.sort_values(by = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP])
	att_numeric = att + '_numeric'
	df[att_numeric] = df[att].astype('category').cat.codes
	att_codes = df[att_numeric].to_array()
	att_vals = df[att].to_arrow().to_pylist()
	att_dict = {key: value.encode('ascii',errors='ignore').decode('ascii').replace(" ","") for key, value in zip(att_codes, att_vals)}
	df[Constants.TEMP_COLUMN_2] = df[att_numeric]
	df = df.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols = [Constants.TEMP_COLUMN_2], outcols= {Constants.TEMP_COLUMN_1: np.uint32})
	num_cases = df[Constants.TARGET_CASE_IDX].nunique()
	df = df.query(Constants.TARGET_CASE_IDX+" == "+Constants.TARGET_PRE_CASE)
	unique_paths_df = df.groupby([Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2]).agg({Constants.TARGET_CASE_IDX: "count"}).reset_index()
	unique_paths_matrix = unique_paths_df[[Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2]].as_gpu_matrix()
	paths_cases_matrix = df[[Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2, Constants.TARGET_CASE_IDX]].as_gpu_matrix()
	paths_feature_cols_matrix = np.zeros((num_cases, unique_paths_matrix.shape[0])).astype("int")
	combinations_kernel.forall(paths_cases_matrix.shape[0])(unique_paths_matrix, paths_cases_matrix, paths_feature_cols_matrix)
	paths_cols_df = cudf.DataFrame.from_records(paths_feature_cols_matrix).reset_index()
	def name_mapper(col_name):
		if col_name == 'index':
			return Constants.TARGET_CASE_IDX
		else:
			code1, code2 = unique_paths_matrix[col_name]
			return att+"_directly@"+att_dict[code1]+"->"+att_dict[code2]
	paths_cols_df = paths_cols_df.rename(columns=name_mapper)
	fea_df = fea_df.merge(paths_cols_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

def select_attribute_eventually_follows_paths(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, column value att_eventually@v1->v2=0 if there is no such eventually-follows path happens in the case, elsewhile = 1
	'''
	df = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP, att]]
	num_cases = df[Constants.TARGET_CASE_IDX].nunique()
	att_numeric = att + '_numeric'
	df[att_numeric] = df[att].astype('category').cat.codes
	att_codes = df[att_numeric].to_array()
	att_vals = df[att].to_arrow().to_pylist()
	att_dict = {key: value.encode('ascii',errors='ignore').decode('ascii').replace(" ","") for key, value in zip(att_codes, att_vals)}
	df = df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	df = df.query(Constants.TARGET_TIMESTAMP + "<" + Constants.TARGET_TIMESTAMP + "_y")
	df = df.sort_values(by = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP])
	unique_paths_df = df.groupby([att_numeric, att_numeric+"_y"]).agg({Constants.TARGET_CASE_IDX: "count"}).reset_index()
	unique_paths_matrix = unique_paths_df[[att_numeric, att_numeric+"_y"]].as_gpu_matrix()
	paths_cases_matrix = df[[att_numeric, att_numeric+"_y", Constants.TARGET_CASE_IDX]].as_gpu_matrix()
	paths_feature_cols_matrix = np.zeros((num_cases, unique_paths_matrix.shape[0])).astype("int")
	combinations_kernel.forall(paths_cases_matrix.shape[0])(unique_paths_matrix, paths_cases_matrix, paths_feature_cols_matrix)
	paths_cols_df = cudf.DataFrame.from_records(paths_feature_cols_matrix).reset_index()
	def name_mapper(col_name):
		if col_name == 'index':
			return Constants.TARGET_CASE_IDX
		else:
			code1, code2 = unique_paths_matrix[col_name]
			return att+"_eventually@"+att_dict[code1]+"->"+att_dict[code2]
	paths_cols_df = paths_cols_df.rename(columns=name_mapper)
	fea_df = fea_df.merge(paths_cols_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

@cuda.jit
def path_durations_kernel(unique_paths_matrix, df_matrix, cases_paths_matrix):
	'''
	This kernel fills cases_paths_matrix with duration of eventually follows paths from the df
	unique_paths_matrix (num_unique_paths x 2): Contains all unique (eventually follows) paths of the df
	df_matrix (num_unique_paths_cases x 4): df in numerical form. Columns: path start value, path end value, case idx, path duration
	cases_paths_matrix (num_cases x num_unique_paths): cases_paths_matrix[i][j] is duration of combination unique_paths_matrix[j] in the case [i]
	'''
	i = cuda.grid(1)
	if i < df_matrix.shape[0]:
		for j in range(unique_paths_matrix.shape[0]):
			if unique_paths_matrix[j][0] == df_matrix[i][0] and unique_paths_matrix[j][1] == df_matrix[i][1]:
				cases_paths_matrix[df_matrix[i][2]][j] = df_matrix[i][3]

def select_attribute_eventually_path_durations(df, fea_df, att):
	'''
	For an attribute att and two values v1, v2, compute duration from first occurence of v1 to last occurence of v2 in the case
	default to 0 if the eventually-follows path v1 -> v2 is not in the case
	'''
	df = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP, att]]
	num_cases = df[Constants.TARGET_CASE_IDX].nunique()
	df[att+"_numeric"] = df[att].astype("category").cat.codes
	att_codes = df[att+"_numeric"].to_array()
	att_vals = df[att].to_arrow().to_pylist()
	att_dict = {key: value.encode('ascii',errors='ignore').decode('ascii').replace(" ","") for key, value in zip(att_codes, att_vals)}
	df = df.merge(df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	df = df.query(Constants.TARGET_TIMESTAMP + "<" + Constants.TARGET_TIMESTAMP + "_y")
	unique_paths_df = df.groupby([att+"_numeric", att+"_numeric_y"]).agg({Constants.TARGET_CASE_IDX: "count"}).reset_index()
	unique_paths_matrix = unique_paths_df[[att+"_numeric", att+"_numeric_y"]].as_gpu_matrix()
	df = df.groupby([att+"_numeric", att+"_numeric_y", Constants.TARGET_CASE_IDX]).agg({Constants.TARGET_TIMESTAMP: "min", Constants.TARGET_TIMESTAMP+"_y": "max"}).reset_index()
	df[Constants.TEMP_COLUMN_1] = df[Constants.TARGET_TIMESTAMP+"_y"] - df[Constants.TARGET_TIMESTAMP]
	df_matrix = df[[att+"_numeric", att+"_numeric_y", Constants.TARGET_CASE_IDX, Constants.TEMP_COLUMN_1]].as_gpu_matrix()
	cases_paths_matrix = np.zeros((num_cases, unique_paths_matrix.shape[0])).astype("int")
	path_durations_kernel.forall(df_matrix.shape[0])(unique_paths_matrix, df_matrix, cases_paths_matrix)
	path_durations_df = cudf.DataFrame.from_records(cases_paths_matrix).reset_index()
	def name_mapper(col_name):
		if col_name == 'index':
			return Constants.TARGET_CASE_IDX
		else:
			code1, code2 = unique_paths_matrix[col_name]
			return att+"_duration_eventually@"+att_dict[code1]+"->"+att_dict[code2]
	path_durations_df = path_durations_df.rename(columns = name_mapper)
	fea_df = fea_df.merge(path_durations_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
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
	Only for string columns
	'''
	df = df[[att1, att2, Constants.TARGET_CASE_IDX]].dropna(subset=[att1, att2])
	df[att1+"_numeric"] = df[att1].astype("category").cat.codes
	df[att2+"_numeric"] = df[att2].astype("category").cat.codes
	att1_vals = df[att1].unique().astype("category").to_arrow().to_pylist()
	att1_vals = [x.encode('ascii',errors='ignore').decode('ascii').replace(" ","") for x in att1_vals]
	att1_codes = df[att1].unique().astype("category").cat.codes
	att1_dict = {att1_codes[i]: att1_vals[i] for i in range(len(att1_vals))}
	att2_vals = df[att2].unique().astype("category").to_arrow().to_pylist()
	att2_vals = [x.encode('ascii',errors='ignore').decode('ascii').replace(" ","") for x in att2_vals]
	att2_codes = df[att2].unique().astype("category").cat.codes
	att2_dict = {att2_codes[i]: att2_vals[i] for i in range(len(att2_vals))}
	df = df[[att1+"_numeric", att2+"_numeric", Constants.TARGET_CASE_IDX]]
	combi_df = df.groupby([att1+"_numeric", att2+"_numeric"]).agg({Constants.TARGET_CASE_IDX: "count"})
	unique_combi_matrix = combi_df.reset_index()[[att1+"_numeric", att2+"_numeric"]].as_gpu_matrix()
	df_matrix = df.as_gpu_matrix()
	combi_case_matrix = np.zeros((df[Constants.TARGET_CASE_IDX].nunique(), unique_combi_matrix.shape[0])).astype("int")
	combinations_kernel.forall(df_matrix.shape[0])(unique_combi_matrix, df_matrix, combi_case_matrix)
	combi_case_df = cudf.DataFrame.from_records(combi_case_matrix).reset_index()
	def name_mapper(col_name):
		if col_name == 'index':
			return Constants.TARGET_CASE_IDX
		else:
			code1, code2 = unique_combi_matrix[col_name]
			return att1+"_and_"+att2+"@"+att1_dict[code1]+"_and_"+att2_dict[code2]
	combi_case_df = combi_case_df.rename(columns=name_mapper)
	fea_df = fea_df.merge(combi_case_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

@cuda.jit
def cases_in_progress_kernel(start_time, end_time, cip):
	'''
	This kernel computes number of cases open during lead time of every case
	start_time, end_time (1D array): start and end time of all cases
	cip (1D array): cip[i] is number of cases open during lead time of case i
	'''
	i = cuda.grid(1)
	if i < len(start_time):
        for j in range(len(end_time)):
            if start_time[i] < end_time[j] and start_time[j] < end_time[i]:
                cip[i] += 1

def select_num_cases_in_progress(df, fea_df, col_name="casesInProgress"):
	'''
	Select number of cases opened at the same time for each case
	'''
	cdf = build_cases_df(df)
	start_time = cdf[Constants.TARGET_TIMESTAMP].to_array()
	end_time = cdf[Constants.TARGET_TIMESTAMP+"_2"].to_array()
	cip = np.zeros(len(start_time))
	cases_in_progress_kernel.forall(len(start_time))(start_time, end_time, cip)
	cdf[col_name] = cip
	cdf[col_name] = cdf[col_name].astype("uint32")
	cdf = cdf[[Constants.TARGET_CASE_IDX, col_name]]
	fea_df = fea_df.merge(cdf, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df

@cuda.jit
def resource_workload_kernel(start_time, end_time, resource_cases, resource_df, workload, resource):
	'''
	This kernel computes the workload of a resource in lead time of all its cases
	start_time, end_time (1D arrays): start and end timestamp of all cases
	resource_cases (1D array): cases that this resource appears in
	resource_df (num_events x 3): df in numerical form: col0 is case idx, col1 is resource idx, col2 is timestamp
	workload (num_cases x num_resources): workload[i][j] = workload of resource j during lead time of case i (if j apepars in i)
	'''
	idx = cuda.grid(1)
	if idx < len(resource_cases):
		case = resource_cases[idx]
		for ev in range(resource_df.shape[0]):
			if start_time[case] <= resource_df[ev][2] and resource_df[ev][2] <= end_time[case]:
				workload[case][resource] += 1

def select_resource_workload_during_case(df, fea_df):
	'''
	Select workload of all resources OF A CASE during timeframe of that case, i.e. number of events performed from start to end in the case
	Value is 0 if resource does not belong to the case.
	'''
	cdf = build_cases_df(df).sort_values(Constants.TARGET_CASE_IDX)
	rsrc = df[Constants.TARGET_RESOURCE].unique()
	rsrc_codes = rsrc.cat.codes
	rsrc_strings = rsrc.to_arrow().to_pylist()
	rsrc_dict = {rsrc_codes[i]: rsrc_strings[i] for i in range(len(rsrc_codes))}
	df = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_RESOURCE_IDX, Constants.TARGET_TIMESTAMP]]
	resources = df[Constants.TARGET_RESOURCE_IDX].unique().to_array()
	cases = cdf[Constants.TARGET_CASE_IDX].to_array()
	start_time = cdf[Constants.TARGET_TIMESTAMP].to_array()
	end_time = cdf[Constants.TARGET_TIMESTAMP+"_2"].to_array()
	workload = np.zeros((len(cdf), resources.shape[0])).astype("uint32")
	for resource in resources:
		resource_df = df[df[Constants.TARGET_RESOURCE_IDX] == resource]
		resource_cases = resource_df[Constants.TARGET_CASE_IDX].unique().to_array()
		resource_df = resource_df.as_gpu_matrix()
		resource_workload_kernel.forall(len(cdf))(start_time, end_time, resource_cases, resource_df, workload, resource)
	workload_df = cudf.DataFrame.from_records(workload).reset_index()
	def name_mapper(col_name):
		if col_name == "index":
			return Constants.TARGET_CASE_IDX
		else:
			return "workload@" + rsrc_dict[col_name].encode('ascii',errors='ignore').decode('ascii').replace(" ","")
	workload_df = workload_df.rename(columns=name_mapper)
	fea_df = fea_df.merge(workload_df, on=[Constants.TARGET_CASE_IDX], how="left", suffixes=('','_y'))
	return fea_df
