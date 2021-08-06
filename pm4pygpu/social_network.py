from pm4pygpu.constants import Constants
from pm4pygpu.dfg import paths_udf, get_frequency_dfg
import numpy as np
from scipy.stats import pearsonr
from pm4pygpu.basic import num_cases
from pm4pygpu.cases_df import build_cases_df

def get_num_cases_of_resource(df):
    '''
    Calculate for each resource the number of cases it is in
    '''
    rdf = df.groupby(Constants.TARGET_RESOURCE).agg({Constants.TARGET_CASE_IDX: "nunique"})
    ret = rdf.to_pandas().to_dict()
    return ret[Constants.TARGET_CASE_IDX]

def handover_graph(df):
    '''
    Return a handover of work metric graph
    Metric: number of handover occurences / (sum(len(case)-1) for all cases in log)
    '''
    df = df.copy()
    dividend = len(df) - num_cases(df)
    rsrc = df[Constants.TARGET_RESOURCE].cat.categories.to_arrow().to_pylist()
    rsrc = {i: rsrc[i] for i in range(len(rsrc))}
    df = df.rename(columns = {Constants.TARGET_RESOURCE_IDX: Constants.TEMP_COLUMN_2})
    df = df = df.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols=[Constants.TEMP_COLUMN_2], outcols={Constants.TEMP_COLUMN_1: np.uint32})
    dfg = df.query(Constants.TARGET_CASE_IDX+" == "+Constants.TARGET_PRE_CASE).groupby([Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2]).agg({Constants.TARGET_EV_IDX: "count"})
    if dividend > 0:
        dfg[Constants.TARGET_EV_IDX] = dfg[Constants.TARGET_EV_IDX] / dividend
    dfg = dfg.to_pandas().to_dict()[Constants.TARGET_EV_IDX]
    dfg = {(str(rsrc[x[0]]), str(rsrc[x[1]])): y for x, y in dfg.items()}
    return dfg

def average_handover_matrix(df):
    '''
    Return a matrix A as DataFrame, whereas A(i,j) indicates the average amount of times i hands over work to j per case
    '''
    df = df.copy()
    case_count = get_num_cases_of_resource(df)
    df = df.sort_values(by = [Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP])
    rsrc_idx = df[Constants.TARGET_RESOURCE_IDX].to_arrow().to_pylist()
    rsrc_str = df[Constants.TARGET_RESOURCE].to_arrow().to_pylist()
    rsrc = {key: value for key, value in zip(rsrc_idx, rsrc_str)}
    ret = df[Constants.TARGET_RESOURCE_IDX].unique().to_frame()
    df[Constants.TEMP_COLUMN_2] = df[Constants.TARGET_RESOURCE_IDX]
    df = df.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols = [Constants.TEMP_COLUMN_2], outcols= {Constants.TEMP_COLUMN_1: np.uint32})
    df = df.query(Constants.TARGET_CASE_IDX+" == "+Constants.TARGET_PRE_CASE)
    for r in rsrc.keys():
        rdf = df[df[Constants.TEMP_COLUMN_1]==r].groupby(Constants.TEMP_COLUMN_2).agg({Constants.TARGET_EV_IDX: "count"}).reset_index()
        rdf = rdf.rename(columns={Constants.TEMP_COLUMN_2: Constants.TARGET_RESOURCE_IDX, Constants.TARGET_EV_IDX: r})  
        ret = ret.merge(rdf, on=[Constants.TARGET_RESOURCE_IDX], how='left', suffixes=("", "_y"))
    ret.index = ret[Constants.TARGET_RESOURCE_IDX]
    ret = ret.drop([Constants.TARGET_RESOURCE_IDX], axis=1)
    ret = ret.fillna(0)
    ret.index = [rsrc[i] for i in ret.index.to_arrow().to_pylist()]
    ret.columns = [rsrc[i] for i in ret.columns]
    for r in ret.columns:
        ret[r] = ret[r] / case_count[r]
    ret = ret.sort_index()
    ret = ret[sorted(ret.columns)]
    return ret.T

def working_together_graph(df):
    '''
    Find a graph indicating the working together metric of any two resources
    Metric = joint cases/num of all cases
    Key (r1, r2) not in graph -> r1 and r2 did not work together (metric = 0)
    '''
    n_cases = num_cases(df)
    rdf = df[[Constants.TARGET_CASE_IDX, Constants.TARGET_RESOURCE]]
    mdf = df.merge(df, on=[Constants.TARGET_CASE_IDX], how='left', suffixes=("","_y"))
    mdf = mdf.groupby([Constants.TARGET_RESOURCE, Constants.TARGET_RESOURCE+"_y"]).agg({Constants.TARGET_CASE_IDX: "nunique"})
    mdf[Constants.TARGET_CASE_IDX] = mdf[Constants.TARGET_CASE_IDX]/n_cases 
    g = mdf.to_pandas().to_dict()[Constants.TARGET_CASE_IDX]
    return g

def similar_activities_graph(df):
    '''
    Return a graph indicating similarity of activities done two resources
    Measured by pearson's correlation coefficient
    No normalization needed by definition of pearson correlation
    '''
    resources = df[Constants.TARGET_RESOURCE].unique().to_arrow().to_pylist()
    acts = df[Constants.TARGET_ACTIVITY_CODE].unique().to_frame()
    for r in resources:
        rdf = df[df[Constants.TARGET_RESOURCE]==r]
        rdf = rdf.groupby([Constants.TARGET_ACTIVITY_CODE]).agg({Constants.TARGET_EV_IDX:"count"}).reset_index()
        rdf = rdf.rename(columns={Constants.TARGET_EV_IDX: r})
        acts = acts.merge(rdf, on=[Constants.TARGET_ACTIVITY_CODE], how='left', suffixes=("", "_y"))
    acts = acts.fillna(0)
    ret = dict()
    for r1 in resources:
        for r2 in resources:
            ret[(r1, r2)] = pearsonr(acts[r1].values.get(), acts[r2].values.get())[0]
    return ret

def subcontracting_graph(df):
    '''
    Subcontracting: r1 -> r2 -> r1 on three successive events in one case
    Return a graph indicating for two resources r1 and r2 the subcontracting metric of r1 r2
    Metric = number of subcontracting instances / number of all cases
    Key (r1, r2) not in graph -> r1 never subcontracted work to r2
    '''
    n_cases = num_cases(df)
    rdf = df.sort_values(by=[Constants.TARGET_CASE_IDX, Constants.TARGET_TIMESTAMP])
    rsrc = df[Constants.TARGET_RESOURCE].cat.categories.to_arrow().to_pylist()
    rsrc = {i: rsrc[i] for i in range(len(rsrc))}
    rdf = rdf.rename(columns={Constants.TARGET_RESOURCE_IDX:Constants.TEMP_COLUMN_2})
    rdf = rdf.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols = [Constants.TEMP_COLUMN_2], outcols= {Constants.TEMP_COLUMN_1: np.uint32})
    rdf = rdf.query(Constants.TARGET_CASE_IDX+"=="+Constants.TARGET_PRE_CASE)
    rdf = rdf.rename(columns={Constants.TEMP_COLUMN_2: Constants.TARGET_RESOURCE_IDX})
    rdf = rdf.rename(columns={Constants.TEMP_COLUMN_1: Constants.TEMP_COLUMN_2})
    rdf = rdf.groupby(Constants.TARGET_CASE_IDX).apply_grouped(paths_udf, incols = [Constants.TEMP_COLUMN_2], outcols= {Constants.TEMP_COLUMN_1: np.uint32})
    min_ev_idxs = rdf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_EV_CASE_MULT_ID: "min"})[Constants.TARGET_EV_CASE_MULT_ID].unique()
    rdf = rdf[~rdf[Constants.TARGET_EV_CASE_MULT_ID].isin(min_ev_idxs)]
    rdf = rdf.query(Constants.TEMP_COLUMN_1+"=="+Constants.TARGET_RESOURCE_IDX)#+" and "+Constants.TEMP_COLUMN_1+" != "+Constants.TEMP_COLUMN_2)
    rdf = rdf.groupby([Constants.TEMP_COLUMN_1, Constants.TEMP_COLUMN_2]).agg({Constants.TARGET_CASE_IDX: "nunique"})
    rdf[Constants.TARGET_CASE_IDX] = rdf[Constants.TARGET_CASE_IDX]/n_cases
    sub_g = rdf.to_pandas().to_dict()[Constants.TARGET_CASE_IDX]
    sub_g = {(str(rsrc[x[0]]), str(rsrc[x[1]])): y for x, y in sub_g.items()}
    return sub_g
