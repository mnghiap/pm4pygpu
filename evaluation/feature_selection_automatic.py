from evaluation.test_logs import test_logs
import cudf
import time
import pandas as pd
import numpy as np
from pm4pygpu.feature_selection import get_automatic_features_df
from pm4pygpu import format
import sys

logs = sys.argv[1:]
parquet = ".parquet"

#cdf = dict()
#pdf = dict()

'''
print("IMPORTING LOGS")
print("===============")
for log in logs:
    cdf[log] = cudf.read_parquet(log + parquet)
    cdf[log] = format.apply(cdf[log])
    pdf[log] = pd.read_parquet(log + parquet)
    columns = list(pdf[log].columns)
    columns = [x.replace("AAA", ":") for x in columns]
    pdf[log].columns = columns
    print(log + ": done!")
'''

from pm4py.objects.log.util.dataframe_utils import automatic_feature_extraction_df as get_automatic_features_df_pm4py


print("AUTOMATIC FEATURE SELECTION")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    cdf = cudf.read_parquet(log + parquet)
    cdf = format.apply(cdf)
    pdf = pd.read_parquet(log + parquet)
    columns = list(pdf.columns)
    columns = [x.replace("AAA", ":") for x in columns]
    pdf.columns = columns
    print("Importing done!")
    t1 = time.time_ns()
    fdf_gpu = get_automatic_features_df(cdf)
    t2 = time.time_ns()
    fdf = get_automatic_features_df_pm4py(pdf)
    t3 = time.time_ns()
    print("PM4PyGPU features df:")
    print(fdf_gpu)
    print("PM4Py features df:")
    print(fdf)
    print(f"PM4PyGPU automatic feature selection time: {t2-t1} ns")
    print(f"PM4Py automatic feature selection time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")