import cudf
import pm4pygpu.format as format
import pm4pygpu.variants as variants
import pm4pygpu.dfg as dfg
import pm4pygpu.efg as efg
import pandas as pd
import pm4py
from evaluation.test_logs import parquet
import pm4py.algo.discovery.dfg.algorithm as pm4pydfg
import time

logs = sys.argv[1:]

cdf = dict()
pdf = dict()

print("IMPORTING LOGS")
print("===============")
for log in logs:
    cdf[log] = cudf.read_parquet(log + parquet)
    cdf[log] = format.apply(cdf[log])
    pdf[log] = pd.read_parquet(log + parquet)
    columns = list(pdf[log].columns)
    columns = [x.replace("AAA", ":") for x in columns]
    pdf[log].columns = columns

from pm4py.algo.discovery.temporal_profile.variants import dataframe as tp
from pm4py.algo.conformance.temporal_profile import algorithm as tpc
def temporal_profile_pm4py(df):
    '''
    Calculate temporal profile on top of pandas df in pm4py
    '''
    temporal_profile = tp.apply(df)
    return tpc.apply(df, temporal_profile, {})

for log in logs:
    if log == "bpic2018":
        print("-----------")
        print("Log: "+ log)
        t1 = time.time_ns()
        tp_cudf = efg.calculate_temporal_profile(cdf[log])
        t2 = time.time_ns()
        tp_pd = temporal_profile_pm4py(pdf[log])
        t3 = time.time_ns()
        print(f"PM4Py temporal profile time: {t2-t1} ns")
        print(f"PM4PyGPU temporal profile time: {t3-t2} ns")
        print(f"Speedup factor {(t3-t2)/(t2-t1)}")
        print("-----------")
