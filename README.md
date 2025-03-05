# pm4pygpu: GPU (CUDF) support for PM4Py

## Introduction

This library provides process mining operations on GPU. Event logs are represented by `cuDF` DataFrames, a library with supporting DataFrames on GPU with APIs similar to pandas. Some operations are implemented using only cuDF APIs. More advanced operations are implemented using Numba CUDA JIT kernels. The main advantage of PM4PyGPU compared to PM4Py is the speedup.

## Installation
To install `pm4pygpu`, clone this repository and then run `python setup.py setup`.

## Example usage
To use `pm4pygpu`, the event log must be loaded using `cuDF` and then formatted by pm4pygpu's formatter. After formatting, the event log contains columns from the `pm4pygpu.Constants` module, allowing pm4pygpu operations to be applied on the log. The following example demonstrates how to compute the duration eventually-follows graph for a given log.

```python
import cudf
import pm4pygpu

# Importing the BPIC 2017 event log in parquet format
df = cudf.read_parquet("bpic2017.parquet")

# Formatting the event log for using PM4PyGPU
df = pm4pygpu.format.apply(df)

# Discover the duration EFG for the "concept:name" attribute in the log
fea_df = df[pm4pygpu.Constants.TARGET_CASE_IDX].unique().to_frame() # Identifying the case attribute
efg = pm4pygpu.feature_selection.select_attribute_eventually_path_durations(df, fea_df, "concept:name")
print(efg)
```

## Speedup evaluation
The branch `evaluation-thesis` provides some evaluation scripts for comparing PM4PyGPU to PM4Py CPU. Instructions to run evaluations are provided in the `README.md` of this branch.
