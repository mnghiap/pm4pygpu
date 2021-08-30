from evaluation.test_logs import test_logs
import cudf
import time
import pandas as pd
import numpy as np
import sys

logs = sys.argv[1:]
parquet = ".parquet"
csv = ".csv"

cdf = dict()
pdf = dict()
cudf_dict = dict()
pd_dict = dict()
cdf_from_dict = dict()
pdf_from_dict = dict()

# Importing from parquet
print("IMPORTING FROM PARQUET")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    cdf[log] = cudf.read_parquet(log + parquet)
    pdf[log] = pd.read_parquet(log + parquet)

print("CONVERTING TO PYDICT")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cudf_dict[log] = {name: col.to_array(fillna="pandas").tolist() for name, col in cdf[log].iteritems()}
    t2 = time.time_ns()
    pd_dict[log] = pdf[log].to_dict()
    t3 = time.time_ns()
    print(f"cuDF conversion time: {t2-t1} ns")
    print(f"pandas conversion time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

print("IMPORTING FROM PYDICT")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cdf_from_dict[log] = cudf.DataFrame(cudf_dict[log])
    t2 = time.time_ns()
    pdf_from_dict[log] = pd.DataFrame(pd_dict[log])
    t3 = time.time_ns()
    print(f"cuDF importing time: {t2-t1} ns")
    print(f"pandas importing time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")