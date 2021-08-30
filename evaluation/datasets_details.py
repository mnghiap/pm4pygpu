import pm4pygpu.format as format
import pm4pygpu.basic as basic
from pm4pygpu.constants import Constants
import cudf
import sys
from evaluation.test_logs import test_logs

logs = test_logs
parquet = ".parquet"

print("DATASETS DETAILS")
print("================")
for log in logs:
    print("Log: " + log)
    df = cudf.read_parquet(log + parquet)
    print(f"Num attributes: {df.columns.size}")
    df = format.apply(df)
    print(f"Num cases:  {basic.num_cases(df)}")
    print(f"Num events: {basic.num_events(df)}")
    print(f"Num variants: {basic.num_variants(df)}")
    print(f"Num activities: {df[Constants.TARGET_ACTIVITY_CODE].nunique()}")
    print(f"Num resources: {df[Constants.TARGET_RESOURCE_IDX].nunique()}")
    print("=================")