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
cdf_csv = dict()
pdf_csv = dict()

# Avoid first time overhead
df = cudf.read_parquet("bpic2017.parquet")

# Importing from parquet
print("IMPORTING FROM PARQUET")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cdf[log] = cudf.read_parquet(log + parquet)
    t2 = time.time_ns()
    pdf[log] = pd.read_parquet(log + parquet)
    t3 = time.time_ns()
    print(f"cuDF importing time: {t2-t1} ns")
    print(f"pandas importing time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

# Exporting to parquet
print("EXPORTING TO PARQUET")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cdf[log].to_parquet(log + "_exported_cudf" + parquet)
    t2 = time.time_ns()
    pdf[log].to_parquet(log + "_exported_pd" + parquet)
    t3 = time.time_ns()
    print(f"cuDF exporting time: {t2-t1} ns")
    print(f"pandas exporting time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

# Exporting to CSV
print("EXPORTING TO CSV")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cdf[log].to_csv(log + "_exported_cudf" + csv)
    t2 = time.time_ns()
    pdf[log].to_csv(log + "_exported_pd" + csv)
    t3 = time.time_ns()
    print(f"cuDF exporting time: {t2-t1} ns")
    print(f"pandas exporting time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

# Importing from CSV
print("IMPORTING FROM CSV")
for log in logs:
    print("-----------")
    print("Log: "+ log)
    t1 = time.time_ns()
    cdf_csv[log] = cudf.read_csv(log + "_exported_cudf" + csv)
    t2 = time.time_ns()
    pdf_csv[log] = pd.read_csv(log + "_exported_pd" + csv)
    t3 = time.time_ns()
    print(f"cuDF importing time: {t2-t1} ns")
    print(f"pandas importing time: {t3-t2} ns")
    print(f"Speedup factor {(t3-t2)/(t2-t1)}")
    print("-----------")

