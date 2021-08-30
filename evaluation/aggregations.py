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

print("FREQUENCY DFG")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    dfg_cudf = dfg.get_frequency_dfg(cdf[log])
    t2 = time.time_ns()
    dfg_pd = pm4pydfg.apply(pdf[log], variant = pm4pydfg.Variants.FREQUENCY)
    t3 = time.time_ns()
    print(f"PM4PyGPU frequency DFG time: {t2-t1} ns")
    print(f"PM4Py frequency DFG time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

print("PERFORMANCE DFG")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    pdfg_cudf = dfg.get_performance_dfg(cdf[log])
    t2 = time.time_ns()
    pdfg_pd = pm4pydfg.apply(pdf[log], variant = pm4pydfg.Variants.PERFORMANCE)
    t3 = time.time_ns()
    print(f"PM4PyGPU performance DFG time: {t2-t1} ns")
    print(f"PM4Py preformance DFG time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

from pm4py.statistics.traces.generic.pandas import case_statistics
def get_variants_pm4py(df):
    '''
    Get variants on top of pandas df in pm4py
    '''
    variants_count = case_statistics.get_variant_statistics(df,
                                          parameters={case_statistics.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      case_statistics.Parameters.ACTIVITY_KEY: "concept:name",
                                                      case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
    return variants_count

print("VARIANTS RETRIEVAL")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    vars_cudf = variants.get_variants(cdf[log])
    t2 = time.time_ns()
    vars_pd = get_variants_pm4py(pdf[log])
    t3 = time.time_ns()
    print(f"PM4PyGPU variants time: {t2-t1} ns")
    print(f"PM4Py variants time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

from pm4py.algo.discovery.temporal_profile.variants import dataframe as tp
from pm4py.algo.conformance.temporal_profile import algorithm as tpc
def temporal_profile_pm4py(df):
    '''
    Calculate temporal profile on top of pandas df in pm4py
    '''
    temporal_profile = tp.apply(df)
    return tpc.apply(df, temporal_profile, {})

print("TEMPORAL PROFILE")
for log in logs:
    if log != "bpic2018":
        print("-----------")
        print("Log: "+ log)
        t1 = time.time_ns()
        tp_cudf = efg.conformance_temporal_profile(cdf[log])
        t2 = time.time_ns()
        tp_pd = temporal_profile_pm4py(pdf[log])
        t3 = time.time_ns()
        print(f"PM4PyGPU temporal profile time: {t2-t1} ns")
        print(f"PM4Py temporal profile time: {t3-t2} ns")
        print(f"Speedup factor {(t3-t2)/(t2-t1)}")
        print("-----------")
