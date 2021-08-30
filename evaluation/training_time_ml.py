from evaluation.test_logs import test_logs
import sys
import cudf
import time
import math
import pandas as pd
import numpy as np
from pm4pygpu.feature_selection import *
from pm4pygpu import format
from pm4pygpu.constants import Constants
from cuml.experimental.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans as skKMeans
from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import DBSCAN as skDBSCAN
from cuml.cluster import DBSCAN as cuDBSCAN
from pm4pygpu.cases_df import get_last_df
from cuml.experimental.preprocessing import KBinsDiscretizer
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as skKNeighborsClassifier

logs = sys.argv[1:]
parquet = ".parquet"

print("TRAINING TIME MACHINE LEARNING ALGORITHMS")
for log in logs:
    print("--------------------")
    print("Log: "+ log)
    df = cudf.read_parquet(log + parquet)
    df = format.apply(df)
    print("Importing done!")
    fea_df = get_automatic_features_df(df)
    if log == 'bpic2019':
        df = df[[Constants.TARGET_TIMESTAMP, Constants.TARGET_TIMESTAMP+"_2", Constants.TARGET_CASE_IDX, Constants.TARGET_RESOURCE_IDX, Constants.TARGET_RESOURCE, Constants.TARGET_EV_IDX, Constants.TARGET_PRE_CASE, 'concept:name', 'org:resource', 'case:ItemCategory']]
    fea_df = select_case_duration(df, fea_df)
    fea_df = select_num_events(df, fea_df)
    fea_df = select_attribute_directly_follows_paths(df, fea_df, "concept:name")
    fea_df = select_time_from_start_of_case(df, fea_df, "concept:name")
    fea_df = select_time_to_end_of_case(df, fea_df, "concept:name")
    fea_df = select_num_cases_in_progress(df, fea_df)
    fea_df = select_resource_workload_during_case(df, fea_df)
    if log == 'bpic2019':
        df = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP, 'concept:name', 'org:resource']]
    fea_df = select_attribute_combinations(df, fea_df, "concept:name", "org:resource")
    fea_df = fea_df.set_index(Constants.TARGET_CASE_IDX).sort_index()
    if log == 'roadtraffic':
        y_roadtraffic = fea_df['totalPaymentAmount']
        fea_df = fea_df.drop(labels=['totalPaymentAmount'], axis=1)
    num_features = len(fea_df.columns)
    scaler = MinMaxScaler()
    scaler.fit(fea_df)
    cu_X = scaler.transform(fea_df)
    sk_X = cu_X.as_matrix()
    print(f"Feature selection done!")
    print(f"Number of features: {num_features}")

    print(f"______ {log}: TRACE CLASSIFICATION ______")
    print(f"Calculating target class for {log}...")
    last_df = get_last_df(df).sort_values(Constants.TARGET_CASE_IDX)
    if log == 'receipt':
        last_df['case:enddate'] = last_df['case:enddate'].fillna(df['time:timestamp'])
        cu_y = (last_df['case:enddate'] < last_df['case:enddate_planned']).astype("int")
    elif log == 'roadtraffic':
        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
        discretizer.fit(y_roadtraffic.to_frame())
        cu_y = discretizer.transform(y_roadtraffic.to_frame())[0]
    elif log == 'bpic2017':
        cu_y = last_df['case:Accepted'].astype("int")
    elif log == 'bpic2017_application':
        cu_y = last_df['case:ApplicationType'].astype("category").cat.codes
    elif log == 'bpic2018':
        cu_y = last_df['case:rejected'].astype("int")
    elif log == 'bpic2019':
        cu_y = last_df['case:ItemCategory'].astype("category").cat.codes
    sk_y = cu_y.to_array()
    print(f"Calculating target class done!")
    print(f"~~~~~~~~~~~~~")
    print(f"====== {log}: RANDOM FOREST CLASSIFIER ======")
    sk_rcf = skRandomForestClassifier(n_estimators=100, random_state=2512, n_jobs=-1)
    cu_rcf = cuRandomForestClassifier(n_estimators=100, random_state=2512)
    t1 = time.time_ns()
    cu_rcf.fit(cu_X, cu_y)
    t2 = time.time_ns()
    print(f"cuML training time: {t2-t1} ns")
    t2 = time.time_ns()
    sk_rcf.fit(sk_X, sk_y)
    t3 = time.time_ns()
    print(f"sklearn training time: {t3-t2} ns")
    print(f"cuML speedup factor: {(t3-t2)/(t2-t1)}")
    print(f"~~~~~~~~~~~~~")
    print(f"~~~~~~~~~~~~~")
    print(f"====== {log}: K-NEAREST NEIGHBORS ======")
    sk_knn = skKNeighborsClassifier(n_neighbors = 2*math.floor(num_features**0.5/2)+1, n_jobs=-1)
    cu_knn = cuKNeighborsClassifier(n_neighbors = 2*math.floor(num_features**0.5/2)+1)
    t1 = time.time_ns()
    cu_knn.fit(cu_X, cu_y)
    t2 = time.time_ns()
    print(f"cuML training time: {t2-t1} ns")
    t2 = time.time_ns()
    sk_knn.fit(sk_X, sk_y)
    t3 = time.time_ns()
    print(f"sklearn training time: {t3-t2} ns")
    print(f"cuML speedup factor: {(t3-t2)/(t2-t1)}")
    print(f"~~~~~~~~~~~~~")

    print(f"______ {log}: TRACE CLUSTERING ______")
    print(f"~~~~~~~~~~~~~")
    print(f"====== {log}: K-MEANS CLUSTERING ======")
    sk_kmeans = skKMeans(n_clusters=5, random_state=2512, n_jobs=-1)
    cu_kmeans = cuKMeans(n_clusters=5, random_state=2512)
    t1 = time.time_ns()
    cu_kmeans.fit(cu_X)
    t2 = time.time_ns()
    sk_kmeans.fit(sk_X)
    t3 = time.time_ns()
    print(f"cuML training time: {t2-t1} ns")
    print(f"sklearn training time: {t3-t2} ns")
    print(f"cuML speedup factor: {(t3-t2)/(t2-t1)}")
    print(f"~~~~~~~~~~~~~")
    print(f"~~~~~~~~~~~~~")
    print(f"====== {log}: DBSCAN CLUSTERING ======")
    sk_dbscan = skDBSCAN(eps=0.1, min_samples=2*num_features, n_jobs=-1)
    cu_dbscan = cuDBSCAN(eps=0.1, min_samples=2*num_features)
    t1 = time.time_ns()
    cu_dbscan.fit(cu_X)
    t2 = time.time_ns()
    print(f"cuML training time: {t2-t1} ns")
    t2 = time.time_ns()
    sk_dbscan.fit(sk_X)
    t3 = time.time_ns()
    print(f"sklearn training time: {t3-t2} ns")
    print(f"cuML speedup factor: {(t3-t2)/(t2-t1)}")
    print(f"~~~~~~~~~~~~~")
    
    print("--------------------")