from pm4pygpu.constants import Constants
import numpy as np
import cudf
import sys
from pm4pygpu import format
from pm4pygpu import feature_selection as fs
import time
from inspect import signature
from evaluation.test_logs import parquet

logs = sys.argv[1:]
parquet = ".parquet"

test_features = [
    fs.select_case_duration,
    fs.select_num_events,
    fs.select_attribute_directly_follows_paths,
    #fs.select_attribute_eventually_follows_paths,
    #fs.select_attribute_eventually_path_durations,
    fs.select_time_from_start_of_case,
    fs.select_time_to_end_of_case,
    fs.select_attribute_combinations,
    fs.select_num_cases_in_progress,
    fs.select_resource_workload_during_case
]

att1 = 'concept:name'
att2 = 'org:resource'

print("ADVANCDE FEATURE EXECUTION TIME AND ADDED COLUMNS")
for log in logs:
    print("==============")
    print("Importing log: " + log)
    df = cudf.read_parquet(log + parquet)
    df = format.apply(df)
    fea_df = df[Constants.TARGET_CASE_IDX].unique().to_frame()
    print("Done!")
    for feature in test_features:
        print("-----------")
        print("Testing log " + log + " on " + feature.__repr__())
        param_list = list(signature(feature).parameters)
        if len(param_list) == 2 or 'col_name' in param_list:
            t1 = time.time_ns()
            dummy = feature(df, fea_df)
            t2 = time.time_ns()
        elif len(param_list) == 3:
            t1 = time.time_ns()
            dummy = feature(df, fea_df, att1)
            t2 = time.time_ns()
        elif len(param_list) == 4:
            t1 = time.time_ns()
            dummy = feature(df, fea_df, att1, att2)
            t2 = time.time_ns()
        print(f"Execution time: {t2-t1} ns")
        print("Features df post execution:")
        print(dummy)
    