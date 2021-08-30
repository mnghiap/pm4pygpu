import cudf
import pm4pygpu.format as format
import pm4pygpu.variants as variants
import pm4pygpu.dfg as dfg
import pm4pygpu.efg as efg
import pandas as pd
import pm4py
from evaluation.test_logs import test_logs, parquet
import time
from pm4pygpu.cases_df import build_cases_df, filter_on_case_perf, get_first_df
from pm4pygpu.start_end_activities import get_start_activities, filter_start_activities
from pm4pygpu.constants import Constants
from pm4pygpu.variants import filter_on_variants

logs = sys.argv[1:]

cdf = dict()
pdf = dict()
duration_param = dict()
start_param = dict()
var_param = dict()

print("IMPORTING LOGS")
print("===============")
for log in logs:
    cdf[log] = cudf.read_parquet(log + parquet)
    cdf[log] = format.apply(cdf[log])
    pdf[log] = pd.read_parquet(log + parquet)
    pdf[log].columns = [x.replace("AAA", ":") for x in pdf[log].columns]
    duration_param[log] = dict()
    start_param[log] = dict()
    var_param[log] = dict()

print("CALCULATING TEST PARAMETERS")
print("===============")
print("CASE DURATION PARAMETERS")
for log in logs:
    cases_df = build_cases_df(cdf[log])
    case_duration = cases_df[Constants.CASE_DURATION].sort_values(ascending = False)
    top80 = case_duration.head(int(case_duration.size*0.8))
    bottom10 = case_duration.tail(int(case_duration.size*0.1))
    duration_param[log][80] = [top80.min(), top80.max()]
    duration_param[log][10] = [bottom10.min(), bottom10.max()]
    print(log + " done!")
print("==============")
print("START ACTIVITIES PARAMETERS")
for log in logs:
    fdf = get_first_df(cdf[log])
    sa = fdf[Constants.TARGET_ACTIVITY].astype("string").value_counts().sort_values(ascending = False).reset_index()['index']
    start_param[log][80] = sa.head(int(case_duration.size*0.8)).to_array().tolist()
    start_param[log][10] = sa.tail(int(case_duration.size*0.1)).to_array().tolist()
    print(log + " done!")
print("==============")
print("VARIANTS PARAMETERS")
for log in logs:
    vdf = variants.get_variants_df(cdf[log])
    vars0 = vdf[Constants.TARGET_ACTIVITY].value_counts().sort_values(ascending=False).reset_index()['index']
    var_param[log][80] = vars0.head(int(case_duration.size*0.8)).to_array().tolist()
    var_param[log][10] = vars0.tail(int(case_duration.size*0.1)).to_array().tolist()
    print(log + " done!")
print("BEGIN TESTING")
print("==============")

from pm4py.algo.filtering.pandas.cases import case_filter
def filter_on_case_perf_pm4py(df, min_perf, max_perf):
    return case_filter.filter_case_performance(df, min_perf, max_perf, parameters={
        case_filter.Parameters.CASE_ID_KEY: "case:concept:name",
        case_filter.Parameters.TIMESTAMP_KEY: "time:timestamp"
    })

print("CASE DURATION TOP 80%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_on_case_perf(cdf[log], duration_param[log][80][0], duration_param[log][80][1])
    t2 = time.time_ns()
    dummy2 = filter_on_case_perf_pm4py(pdf[log], duration_param[log][80][0], duration_param[log][80][1])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

print("CASE DURATION BOTTOM 10%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_on_case_perf(cdf[log], duration_param[log][10][0], duration_param[log][10][1])
    t2 = time.time_ns()
    dummy2 = filter_on_case_perf_pm4py(pdf[log], duration_param[log][10][0], duration_param[log][10][1])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

from pm4py.algo.filtering.pandas.start_activities import start_activities_filter
def filter_start_activities_pm4py(df, list_act):
    return  start_activities_filter.apply(df, list_act, parameters={
        start_activities_filter.Parameters.CASE_ID_KEY: "case:concept:name",
        start_activities_filter.Parameters.ACTIVITY_KEY: "concept:name"
    })

print("START ACTIVITIES TOP 80%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_start_activities(cdf[log], start_param[log][80])
    t2 = time.time_ns()
    dummy2 = filter_start_activities_pm4py(pdf[log], start_param[log][80])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

print("START ACTIVITIES BOTTOM 10%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_start_activities(cdf[log], start_param[log][10])
    t2 = time.time_ns()
    dummy2 = filter_start_activities_pm4py(pdf[log], start_param[log][10])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

from pm4py.algo.filtering.pandas.variants import variants_filter
def filter_on_variants_pm4py(df, variants):
    return variants_filter.apply(df, variants, parameters={
        variants_filter.Parameters.CASE_ID_KEY: "case:concept:name",
        variants_filter.Parameters.ACTIVITY_KEY: "concept:name"
    })

print("VARIANTS TOP 80%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_on_variants(cdf[log], var_param[log][80])
    t2 = time.time_ns()
    dummy2 = filter_on_variants_pm4py(pdf[log], var_param[log][80])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

print("VARIANTS BOTTOM 10%")
for log in logs:
    print("-----------")
    print("Log: " + log)
    t1 = time.time_ns()
    dummy1 = filter_on_variants(cdf[log], var_param[log][10])
    t2 = time.time_ns()
    dummy2 = filter_on_variants_pm4py(pdf[log], var_param[log][10])
    t3 = time.time_ns()
    print(f"PM4PyGPU filtering time: {t2-t1} ns")
    print(f"PM4Py filtering time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")
