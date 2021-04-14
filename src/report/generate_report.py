import networkx as nx
import pandas as pd
from loguru import logger
import os
from tqdm import tqdm
import build_graph
import misc.constants as constants
import misc.graph_utils as gu

if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('results/reports'):
    os.makedirs('results/reports')

# load the data
predictions_file = "results/predictions/predictions_RF_2020_09_02_07_50_01.csv"
logger.info(f"Reading predictions from {predictions_file}")
predictions = pd.read_csv(predictions_file)

# select the positive predictions
logger.info("Selecting positive predictions")
positive_predictions = predictions[predictions["prediction"] == 1]
logger.info(f"Selected {len(positive_predictions)} positive predictions")

# save to disk
positive_predictions_file = "results/reports/positive_predictions.csv"
logger.info(f"Saving positive predictions to {positive_predictions_file}")
positive_predictions.to_csv(positive_predictions_file,index=False)

# create a graph from edges
prediction_graph = nx.DiGraph()
logger.info("Generating positive predictions graph")
for index, row in tqdm(positive_predictions.iterrows()):
    source_node = row["source"]
    target_node = row["target"]
    prediction_graph.add_edge(source_node, target_node)

# read the original graph
graph_type = "no_sites"
original_graph = build_graph.build_graph(constants.INPUT_FILE,
                        f"data/output/iptmnet_knowledge_graph_{graph_type}",
                        graph_type=graph_type,
                        num_rows=constants.PARAMS["num_rows"]
                    )
logger.info(f"Original graph {len(original_graph.nodes())}")

####
# For kinases that had less than five substrates, for how many where we able to identify new substrates?
####
num_substrates = 5
logger.info(f"Getting kinases with <= {num_substrates} substrates")
kinases_lt_5_sub = gu.get_kinases_with_substrates_lt_or_eq_to(original_graph,num_substrates)
logger.info(f"Found {len(kinases_lt_5_sub)} kinases with <= {num_substrates} substrates")

logger.info("Checking for presence of these kinases in positive graph")
common_kinases = gu.find_common_proteins(prediction_graph,kinases_lt_5_sub)
logger.info(f"Found {len(common_kinases)} kinases with <= {num_substrates} substrates in original graph for which prediction where made.")

####
#  Sort common kinases by number of substrates
####
out_degree_predicted = list(prediction_graph.out_degree(common_kinases))

top_kinases_df = pd.DataFrame(out_degree_predicted, columns=["kinase","n_substrates"])
top_kinases_df = top_kinases_df.sort_values(by=["n_substrates"],ascending=False)

# save to disk
top_kinases_out_file = "results/reports/top_kinases.csv"
logger.info(f"Saving the top kinases to {top_kinases_out_file}")
top_kinases_df.to_csv(top_kinases_out_file, index=False)

####
#  Create a list of kinases and their substrates
####
top_kinases_sub = []
for index, row in top_kinases_df.iterrows():
    # get the substrates for this kinase
    kinase = row["kinase"]
    substrates = gu.get_substrates_for(prediction_graph,kinase)
    data = {
        "kinase": kinase,
        "substrates": ",".join(substrates)
    }
    top_kinases_sub.append(data)

top_kinases_sub_pd = pd.DataFrame.from_dict(top_kinases_sub)

# save to disk
top_kinases_sub_out_file = "results/reports/top_kinases_with_substrates.csv"
logger.info(f"Saving the top kinases with substrates to {top_kinases_out_file}")
top_kinases_sub_pd.to_csv(top_kinases_sub_out_file, index=False)

###
# For common kinases compare the number or original substrates vs predicted substrates
###
out_degree_original = original_graph.out_degree(common_kinases)

# merge out degree predicted and out degree original
merged_out_degree = []
for idx, out_degree in enumerate(out_degree_original):
    predicted_degree = out_degree_predicted[idx][1]
    merged_degree = (out_degree[0],out_degree[1],predicted_degree)
    merged_out_degree.append(merged_degree)

merged_out_degree_df = pd.DataFrame(merged_out_degree,columns=["kinase","n_substrates_original","n_substrates_predicted"])
merged_out_degree_df = merged_out_degree_df.sort_values(by=["n_substrates_predicted"],ascending=False)

# save to disk
merged_degree_out_file = "results/reports/top_kinases_substrates_number_compare.csv"
logger.info(f"Saving merged degrees to {merged_degree_out_file}")
merged_out_degree_df.to_csv(merged_degree_out_file, index=False)


###
# Save the predictions graph as graphml
###
prediction_graph_out_file = "results/reports/predictions_graph.graphml"
logger.info(f"Saving prediction graph to - {prediction_graph_out_file}")
nx.write_graphml(prediction_graph,f"{prediction_graph_out_file}")


####
# Merge both the graph to produce a new graph for visualizing the results
####
logger.info("Tagging graphs")
prediction_graph = gu.tag_edges(prediction_graph,"predicted","true")
original_graph = gu.tag_edges(original_graph,"predicted","false")

# merge the graph
logger.info("Merging graphs")
merged_graph = nx.compose(prediction_graph,original_graph)

# saving the graph
merged_out_file = "results/reports/merged_graph.graphml"
logger.info(f"Writing graph to {merged_out_file}")
nx.write_graphml(merged_graph,f"{merged_out_file}")

###
# Identify kinases for IDG protein list
###
idg_proteins_df = pd.read_csv("data/input/idg_protein_list.csv")

idg_kinases_list = []
for index, row in idg_proteins_df.iterrows():
    protein = row["Substrates"]
    found_edges = positive_predictions[positive_predictions["target"] == protein]
    found_kinases = found_edges["source"].tolist()
    data = {
        "substrate": protein,
        "kinases": ",".join(found_kinases)
    }
    idg_kinases_list.append(data)

idg_kinases_df = pd.DataFrame.from_dict(idg_kinases_list)
idg_kinaes_out_file = "results/reports/idg_kinases.csv"
logger.info(f"Writing IDG kinases to {idg_kinaes_out_file}")
idg_kinases_df.to_csv(idg_kinaes_out_file,index=False)

# Get the number of IDG proteins in iPTMnet
idg_proteins_common = gu.find_common_proteins(original_graph,idg_proteins_df["Substrates"].tolist())
logger.info(f"{len(idg_proteins_common)} proteins from IDG list present in iPTMnet")

###
# Create a subgraph of IDG protein with identified kinases
###

# get the idg proteins with atleast one predicted kinase
idg_proteins_with_kinases_df = idg_kinases_df[idg_kinases_df["kinases"] != ""]

# extract all these proteins to a list
proteins_of_interest = []
for index, row in idg_proteins_with_kinases_df.iterrows():
    proteins_of_interest.append(row["substrate"])
    kinases = row["kinases"].split(",")
    proteins_of_interest.extend(kinases)

logger.info(f"Found {len(proteins_of_interest)} proteins of interest.")
logger.info(f"Regex for proteins of interest : {'|'.join(proteins_of_interest)}")

logger.info("Done")