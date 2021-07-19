from pm4pygpu.constants import Constants

# Generally, positive means cases conforming to the name of the method
def four_eyes_principle(df, act1, act2, positive=False):
    '''
    Four eyes principle: act1 and act2 should not be performed by the same resource
    return cases satisfying 4 eyes if positive is true, else return cases violating the principle
    '''
    adf = df[df[Constants.TARGET_ACTIVITY].astype('string').isin([act1, act2])]
    adf = adf.groupby([Constants.TARGET_CASE_IDX, Constants.TARGET_RESOURCE_IDX]).agg({Constants.TARGET_ACTIVITY_CODE: "nunique"}).reset_index()
    adf = adf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY_CODE: "max"}).reset_index()
    violating_case_idxs = adf[adf[Constants.TARGET_ACTIVITY_CODE] == 2][Constants.TARGET_CASE_IDX].unique()
    if positive:
        return df[~df[Constants.TARGET_CASE_IDX].isin(violating_case_idxs)]
    else:
        return df[df[Constants.TARGET_CASE_IDX].isin(violating_case_idxs)]

def activity_from_different_persons(df, act, positive=True):
    '''
    If positive is true, return cases where act is done by different resources
    Elsewhile return other cases
    '''
    adf = df[df[Constants.TARGET_ACTIVITY].astype('string').isin([act])]
    adf = adf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_RESOURCE_IDX: "nunique"}).reset_index()
    different_persons_case_idxs = adf[adf[Constants.TARGET_RESOURCE_IDX] >= 2][Constants.TARGET_CASE_IDX].unique()
    if positive:
        return df[df[Constants.TARGET_CASE_IDX].isin(different_persons_case_idxs)]
    else:
        return df[~df[Constants.TARGET_CASE_IDX].isin(different_persons_case_idxs)]

def never_together(df, act1, act2, positive=False):
    '''
    Ideally, act1 and act2 should not happen together in a case.
    If positive is true, return cases where act1 and act2 didn't happen together
    Elsewhile return cases where they instead happen together
    '''
    adf = df[df[Constants.TARGET_ACTIVITY].astype('string').isin([act1, act2])]
    adf = adf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY_CODE: "nunique"}).reset_index()
    violating_case_idxs = adf[adf[Constants.TARGET_ACTIVITY_CODE] == 2][Constants.TARGET_CASE_IDX].unique()
    if positive:
        return df[~df[Constants.TARGET_CASE_IDX].isin(violating_case_idxs)]
    else:
        return df[df[Constants.TARGET_CASE_IDX].isin(violating_case_idxs)]

def equivalence(df, act1, act2, positive=True):
    '''
    Two activities act1 and act2 are equivalent if they happen equally often in the case
    If positive is true, return cases where act1 and act2 are equivalence
    Elsewhile, return cases where numbers of event with act1 and act2 differ
    '''
    adf = df[df[Constants.TARGET_ACTIVITY].astype('string').isin([act1, act2])]
    adf[Constants.TEMP_COLUMN_1] = 0
    adf = adf.groupby([Constants.TARGET_CASE_IDX, Constants.TARGET_ACTIVITY_CODE]).agg({Constants.TEMP_COLUMN_1: "count"}).reset_index()
    adf[Constants.TEMP_COLUMN_2] = adf[Constants.TEMP_COLUMN_1]
    adf = adf.groupby(Constants.TARGET_CASE_IDX).agg({Constants.TARGET_ACTIVITY_CODE: "nunique", Constants.TEMP_COLUMN_1:"max", Constants.TEMP_COLUMN_2:"min"}).reset_index()
    adf[Constants.TEMP_COLUMN_1] = adf[Constants.TEMP_COLUMN_1] - adf[Constants.TEMP_COLUMN_2]
    adf = adf.query(Constants.TARGET_ACTIVITY_CODE+"!=2 or "+Constants.TEMP_COLUMN_1+"!=0")
    not_equivalence_case_idxs = adf[Constants.TARGET_CASE_IDX].unique()
    if positive:
        return df[~df[Constants.TARGET_CASE_IDX].isin(not_equivalence_case_idxs)]
    else:
        return df[df[Constants.TARGET_CASE_IDX].isin(not_equivalence_case_idxs)]
