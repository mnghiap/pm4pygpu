from pm4pygpu.constants import Constants
from pm4pygpu.variants import get_variants_df

def num_events(df):
	return int(len(df))

def num_cases(df):
	return int(df[Constants.TARGET_CASE_IDX].nunique())

def num_variants(df):
	var_df = get_variants_df(df)[Constants.TARGET_ACTIVITY].unique()
	return int(len(var_df))

def get_events_of_case(df, case_id):
	df2 = df.query(Constants.TARGET_CASE_IDX + " == " + str(case_id))
	df2["@@classifier"] = df2[Constants.TARGET_ACTIVITY]
	cols = list(df2.columns)
	cols = [x for x in cols if not x.startswith("custom_column")]
	df2 = df2[cols]
	ret = df2.to_pandas().to_dict("r")
	return ret

def get_csv(df):
	from io import BytesIO
	f = BytesIO()
	df.to_csv(f)
	return f.getvalue()

def get_xes(df):
	from pm4py.objects.log.log import EventStream
	from pm4py.objects.conversion.log import converter
	from pm4py.objects.log.exporter.xes.exporter import serialize
	cols = [x for x in df.columns if not x == "index"]
	stream = df[cols].to_arrow().to_pydict()
	list_eve = EventStream()
	cols = list(stream.keys())
	for i in range(len(stream[cols[0]])):
		ev = {}
		for k in cols:
			ev[k] = stream[k][i]
		list_eve.append(ev)
	list_eve = converter.apply(list_eve)
	return serialize(list_eve)

def calculate_fps(df):
	from pm4pygpu import dfg, start_end_activities, attributes
	dfgg = dfg.get_frequency_dfg(df)
	start_activities = start_end_activities.get_start_activities(df)
	end_activities = start_end_activities.get_end_activities(df)
	activities = attributes.get_attribute_values(df)
	parallel = {(x, y) for (x, y) in dfgg if (y, x) in dfgg}
	sequence = {(x, y) for (x, y) in dfgg if not (y, x) in dfgg}
	return {"dfg": dfgg, "start_activities": start_activities, "end_activities": end_activities, "activities": activities, "parallel": parallel, "sequence": sequence}
