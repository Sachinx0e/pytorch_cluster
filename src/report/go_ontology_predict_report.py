import pandas as pd
import data.data_utils as du
from p_tqdm import p_map
from loguru import logger
import networkx as nx
import misc.graph_utils as gu
import numpy as np
from multiprocessing import Pool
import ray
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score,roc_auc_score
from tqdm import tqdm

def report_data():
    # load the predictions for experimental relations
    predicted_df = pd.read_csv("results/predictions/go_experiment/idg_predictions_rf.csv")

    # load the experimental relations
    experimental_edges = du.get_experimental_edges()
    logger.info(f"Number of experimental edges with loops: {len(experimental_edges)}")

    # remove self loops
    experimental_edges_wo_loops = []
    for edge in experimental_edges:
        if edge[0] != edge[1]:
            experimental_edges_wo_loops.append(edge)
    
    logger.info(f"Number of experimental edges without loops: {len(experimental_edges_wo_loops)}")

    # get the protein with least number of edges in network
    iptmnet_graph = nx.read_gml("data/input/iptmnet_graph_cleaned_directed.gml")
    idg_proteins = pd.read_csv("data/input/idg_protein_list.csv")["Substrates"].tolist()
    
    # get degrees
    logger.info("Getting node degrees")
    degrees = []
    for idg_protein in idg_proteins:
        degree = iptmnet_graph.degree(idg_protein)
        if isinstance(degree,int) == False:
            degree = 0
        degrees.append((idg_protein,degree))

    degrees.sort(key=lambda x: x[1])
    logger.info(f"Protein degrees: {degrees}")

    # get the proteins with atleast one interaction in iptmnet
    proteins_with_least_interactions = list(map(lambda x: x[0],filter(lambda x: x[1] == 1,degrees)))
    logger.info(f"Proteins with least interactions: {proteins_with_least_interactions}")

    # for proteins with at least one interaction get all substrates
    substrates_df = _get_predicted_substrates(proteins_with_least_interactions,predicted_df,0.80)
    substrates_df["count"] = substrates_df["substrates"].apply(lambda x : len(x))
    substrates_df = substrates_df.sort_values(by=["count"],ascending=False)
    print(substrates_df)

    # save these substrates
    substrates_df.to_csv("results/reports/go_experiment/understudied_predicted_substrates.csv",index=False)

    # find number of predicted at 0.5
    thresholds = [0.5,0.6,0.7,0.8,0.9]

    def _get_predicted_tuples(threshold):
        predicted_at_threshold = []
        for edge in experimental_edges_wo_loops:
            is_present = _check_if_present(edge,predicted_df,threshold)
            if is_present:
                predicted_at_threshold.append(edge)
        return ((str(threshold),len(predicted_at_threshold),predicted_at_threshold))

    predicted_at_thresholds_tuples = p_map(lambda x: _get_predicted_tuples(x),thresholds)
    predicted_at_thresholds_df = pd.DataFrame(predicted_at_thresholds_tuples,columns=["threshold","count","predicted_edges"])
    print(predicted_at_thresholds_df)

    # save
    predicted_at_thresholds_df.to_csv("results/reports/go_experiment/predicted_at_threshold.csv",index=False)

def report_figures():
    # load the experimental relations
    experimental_edges = du.get_experimental_edges()
    logger.info(f"Number of experimental edges with loops: {len(experimental_edges)}")

    # remove self loops
    experimental_edges_wo_loops = []
    for edge in experimental_edges:
        if edge[0] != edge[1]:
            experimental_edges_wo_loops.append(edge)
    
    logger.info(f"Number of experimental edges without loops: {len(experimental_edges_wo_loops)}")

    # load the predicted at threshold
    predicted_at_threshold = pd.read_csv("results/reports/go_experiment/predicted_at_threshold.csv")

    thresholds = predicted_at_threshold["threshold"].tolist()
    counts = predicted_at_threshold["count"].tolist()
    total = predicted_at_threshold["count"].apply(lambda x: len(experimental_edges_wo_loops)-x).tolist()

    # show number of predicted experimental edges at different thresholds
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Bar(name='Predicted', x=thresholds, y=counts,marker_color="rgba(18, 123, 227,255)"),
        go.Bar(name='Actual', x=thresholds, y=total,marker_color="rgba(18, 123, 227,50)",opacity=0.2)
    ])

    # Change the bar mode
    fig.update_layout(barmode='stack',xaxis = dict(
        tickmode = 'linear',
        ticks="outside",
        ticklen=10,
        tick0 = 0.5,
        dtick = 0.1
    ),
    yaxis = dict(
        ticks="outside",
        ticklen=10
    ),
    font_size=18,
    title="Number of predicted experimental interactions at varying thresholds",
    width=1024,height=768)
    
    fig.write_image("results/reports/go_experiment/prediced_percentage.png")

    # pathway analysis for Q8N5S9
    

def calculate_metric_for_experimental():
    # load the predictions for experimental relations
    predicted_df = pd.read_csv("results/predictions/go_experiment/idg_predictions_rf.csv")

    # load the experimental relations
    experimental_edges = du.get_experimental_edges()
    logger.info(f"Number of experimental edges with loops: {len(experimental_edges)}")

    # remove self loops
    experimental_edges_wo_loops = []
    for edge in experimental_edges:
        if edge[0] != edge[1]:
            experimental_edges_wo_loops.append(edge)
    
    logger.info(f"Number of experimental edges without loops: {len(experimental_edges_wo_loops)}")

    # sample equal number of negative edges for every experimental edge for every protein in experimental set
    negative_edges =  gu.sample_negative_edges(experimental_edges_wo_loops)


    # create ground truth array
    positive_edges_labels = np.ones((len(experimental_edges_wo_loops)))
    negative_edges_labels = np.zeros(len(negative_edges))
    
    ground_truth_edges = experimental_edges_wo_loops + negative_edges
    ground_truth_labels = np.concatenate([positive_edges_labels,negative_edges_labels])

    # create the predicted labels
    ray.init(include_dashboard=False)
    
    predicted_id = ray.put(predicted_df)

    thresholds = [0.5,0.6,0.7,0.8,0.9]
    
    scores = []

    for threshold in tqdm(thresholds):
        predicted_labels_list = [get_label.remote(edge,predicted_id,threshold) for edge in ground_truth_edges]
        predicted_labels_list = ray.get(predicted_labels_list)
        predicted_labels = np.array(predicted_labels_list)
        
        f1 = f1_score(ground_truth_labels,predicted_labels)
        precision = precision_score(ground_truth_labels,predicted_labels)
        recall = recall_score(ground_truth_labels,predicted_labels)
        au_pr = average_precision_score(ground_truth_labels,predicted_labels)
        roc_auc = roc_auc_score(ground_truth_labels,predicted_labels)
        scores.append((threshold,f1,precision,recall,au_pr,roc_auc))

    scores_df = pd.DataFrame(data=scores,columns=["threshold","f1","precision","recall","au_pr","roc_auc"])  
    scores_df.to_csv("results/reports/go_experiment/scores_experimental.csv",index=False)


@ray.remote
def get_label(edge,predicted_df,threshold):
        is_present = _check_if_present(edge,predicted_df,threshold)
        if is_present:
            return 1
        else:
            return 0
    

def _get_predicted_substrates(kinases,predicted_df,threshold):
    predicted_rows = predicted_df[(predicted_df["source"].isin(kinases)) & (predicted_df["prediction"] >= threshold ) ]
    substrates_tuples = []
    for kinase in kinases:
        substrates = predicted_rows[predicted_rows["source"] == kinase]["target"].tolist()
        substrates_tuples.append((kinase,substrates))

    substrates_pd = pd.DataFrame(substrates_tuples,columns=["kinases","substrates"])
    return substrates_pd

def _check_if_present(experimental_edge,predicted_df,threshold):
    # get the predicted edge
    predicted_edges = predicted_df[(predicted_df["source"] == experimental_edge[0]) & (predicted_df["target"] == experimental_edge[1])]
    if len(predicted_edges) > 0:
        predicted_edge = predicted_edges.iloc[0]
        probability = predicted_edge["prediction"]
        if probability >= threshold:
            return True
        else:
            return False
    else:
        raise Exception(f"No row for : {experimental_edge}")