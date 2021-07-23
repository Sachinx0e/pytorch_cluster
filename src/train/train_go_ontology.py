"""
Train the final models using the best hyper parameters from the GO ontology experiment.
"""

import data.go_embedding_torch as go_embedding
import data.iptmnet_embedder as iptmnet_embedder
import numpy as np
from loguru import logger
import networkx as nx
import train.train_random_forest_supervised as train_rf
import data.feature_builder as feature_builder
import json
from pathlib import Path
import pandas as pd
import torch
import data.data_utils as du
from misc import utils

def train():
    # read the best result from go experiments directory
    results_dir = "results/go_experiment"
    best_experiment_file = utils.get_best_experiment(results_dir)
    results_df = pd.read_csv(best_experiment_file)

    # sort 
    results_df = results_df.sort_values(by=["au_pr","au_roc"],ascending=False)

    # get the first row
    best_result = results_df.iloc[0]

    # build the config
    best_result_dict = best_result.to_dict()

    # extract the config
    config = utils.extract_config(best_result_dict)

    logger.info(config)

    seed = int(config["seed"])
    
    # set seed values
    torch.manual_seed(seed)
    np.random.seed(seed)

    # data
    data_folder="/home/sachin/Projects/kinome-explore/data"
    go_file = f"{data_folder}/input/ontology/go.obo"
    iptmnet_graph_file = f"{data_folder}/input/iptmnet_graph_cleaned_directed.gml"
    experimental_edges_file = f"{data_folder}/input/dark_kinase_substrates.txt"
    parent_mapping_file = f"{data_folder}/output/go_parent_mapping.csv"
    num_epochs=10001

    # build go embedding
    go_embedding_df = go_embedding.build(
        go_file=go_file,
        seed=seed,
        embedding_dim=int(config["go_node2vec_embedding_dim"]),
        walk_length=int(config["go_node2vec_walk_length"]),
        context_size=int(config["go_node2vec_context_size"]),
        walks_per_node=int(config["go_node2vec_walks_per_node"]),
        p=config["go_node2vec_p"],
        q=config["go_node2vec_q"],
        num_negative_samples=int(config["go_node2vec_num_negative_samples"]),
        save=True,
        num_epochs=num_epochs,
        show_progress=True
    )

    # load the graph file
    logger.info(f"Loading data from {iptmnet_graph_file}")
    iptmnet_graph = nx.read_gml(iptmnet_graph_file)

    go_term_type = config["go_term_types"]
    if go_term_type != "none":
        # create features df
        logger.info("Building features")
        features_df = feature_builder.build_features_only_go_term_type(go_term_type,
                                                                    graph=iptmnet_graph,
                                                                    go_embeddings_df=go_embedding_df,
                                                                    parent_mapping_file=parent_mapping_file,
                                                                    binary_operator=config["fb_binary_operator"],
                                                                    save=False)
    else:
        # create emtpy features df
        proteins = list(iptmnet_graph.nodes())
        features_df = pd.DataFrame(data=np.zeros((len(proteins),2)))
        features_df.insert(0,"protein",proteins)

    # build iptmnet embedding
    iptmnet_embedding_df = iptmnet_embedder.embed(
        iptmnet_graph=iptmnet_graph,
        embedding_dim=int(config["iptmnet_node2vec_embedding_dim"]),
        walk_length=int(config["iptmnet_node2vec_walk_length"]),
        context_size=int(config["iptmnet_node2vec_context_size"]),
        walks_per_node=int(config["iptmnet_node2vec_walks_per_node"]),
        p=config["iptmnet_node2vec_p"],
        q=config["iptmnet_node2vec_q"],
        num_negative_samples=int(config["go_node2vec_num_negative_samples"]),
        save=True,
        num_epochs=num_epochs,
        show_progress=True
    )

    # load experimental edges
    experimental_edges = du.get_experimental_edges(experimental_edges_file=experimental_edges_file)

    # train random forest model
    scores = train_rf.train(
        iptmnet_graph=iptmnet_graph,
        experimental_edges=experimental_edges,
        iptmnet_embedding_df=iptmnet_embedding_df,
        features_df=features_df,
        max_depth=config["rf_max_depth"],
        max_features=config["rf_max_features"],
        min_samples_split=int(config["rf_min_samples_split"]),
        min_samples_leaf=int(config["rf_min_samples_leaf"]),
        criterion=config["rf_criterion"],
        edge_operator=config["rf_edge_operator"],
        plots=True,
        show_progress=True,
        save=True,
        perform_testing=True
    )

    print(scores)


