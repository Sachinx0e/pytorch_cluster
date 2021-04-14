import pandas as pd

def report():
    # load event data for humans
    ptm_events = pd.read_csv("data/input/MV_EVENT_human_only.tsv",sep="\t")

    print(ptm_events.head())

    # group by event name
    ptm_events_subset = ptm_events[["iptm_event_id","event_name"]]
    ptm_event_by_type = ptm_events_subset.groupby(by=["event_name"]).count().reset_index().sort_values(by=["iptm_event_id"],ascending=False)
    ptm_event_by_type = ptm_event_by_type.rename(columns={"iptm_event_id":"count"})
    
    # calculate a total
    total = ptm_event_by_type["count"].sum()
    ptm_event_by_type = ptm_event_by_type.append({"event_name":"Total","count":total},ignore_index=True)

    # calculate percentage
    ptm_event_by_type["percentage"] = ptm_event_by_type["count"].apply(lambda x: (x/total)*100)

    print(ptm_event_by_type)

    # save the data
    ptm_event_by_type.to_csv("results/reports/ptm_type_count.csv",index=False)