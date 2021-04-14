"""
Perform an experiment to determine if go ontology contributes anything
to the overall performance.
"""

import data.go_embedding_torch as go_embedding
import data.iptmnet_embedder as iptmnet_embedder
import numpy as np
from ray import tune
from ray.tune.utils.log import Verbosity
from loguru import logger
import networkx as nx
import train.train_random_forest_supervised as train_rf
import data.feature_builder as feature_builder
import json
from pathlib import Path
import pandas as pd
import torch
import data.data_utils as du
import multiprocessing
import ray

def get_config():
    return {
        "go_term_types": tune.grid_search(["all","bp","cc","mp","none"]),
        "fb_binary_operator": tune.choice(["hadamard","avg","cat"]),
        "rf_edge_operator": tune.choice(["hadamard","avg","cat"]),
        "seed": 20,
        "go_node2vec_embedding_dim": tune.qrandint(32,128,16),
        "go_node2vec_walk_length": tune.qrandint(50,100,10),
        "go_node2vec_context_size": tune.qrandint(10,20,5),
        "go_node2vec_walks_per_node": tune.qrandint(10,20,10),
        "go_node2vec_p": tune.quniform(0.1,1.0,0.1),
        "go_node2vec_q": tune.quniform(0.1,1.0,0.1),
        "go_node2vec_num_negative_samples": tune.randint(1,4),
        "iptmnet_node2vec_embedding_dim": tune.qrandint(32,128,16),
        "iptmnet_node2vec_walk_length": tune.qrandint(50,100,10),
        "iptmnet_node2vec_context_size": tune.qrandint(10,20,5),
        "iptmnet_node2vec_walks_per_node": tune.qrandint(10,20,10),
        "iptmnet_node2vec_p": tune.quniform(0.1,1.0,0.1),
        "iptmnet_node2vec_q": tune.quniform(0.1,1.0,0.1),
        "iptmnet_node2vec_num_negative_samples": tune.randint(1,4),
        "rf_max_depth": tune.quniform(0.1,1.0,0.1),
        "rf_criterion": tune.choice(["gini","entropy"]),
        "rf_max_features": tune.choice(["auto","sqrt", "log2"]),
        "rf_min_samples_split": tune.qrandint(2,100,1),
        "rf_min_samples_leaf": tune.qrandint(2,10,1),
    }

def start():
    # set seed values
    torch.manual_seed(20)
    np.random.seed(20)

    # get config
    config = get_config()
    
    # define and run tune
    ray.init(num_gpus=torch.cuda.device_count())
    analysis = tune.run(
        _execute_experiment,
        config=config,
        num_samples=10,
        metric="au_roc",
        mode="max",
        verbose=Verbosity.V0_MINIMAL,
        resources_per_trial={"cpu":6, "gpu": 1},
        raise_on_failed_trial=False,
    )

    # save results
    df_results = analysis.results_df

    # create result dir if not exists
    from datetime import datetime
    date_serialized = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    result_dir = "/home/sachin/Projects/iptmnet_knowledge_graph/results/go_experiment"
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # write the results
    result_file = f"{result_dir}/result_{date_serialized}.csv"
    df_results.to_csv(result_file,index=False)
    logger.info(f"Results written to {result_file}")

def _execute_experiment(config):
    torch.cuda.empty_cache()

    logger.info(f"Config : {config}")

    # data
    data_folder="/home/sachin/Projects/kinome_explore/data"
    go_file = f"{data_folder}/input/ontology/go.obo"
    iptmnet_graph_file = f"{data_folder}/input/iptmnet_graph_cleaned_directed.gml"
    experimental_edges_file = f"{data_folder}/input/dark_kinase_substrates.txt"
    parent_mapping_file = f"{data_folder}/output/go_parent_mapping.csv"
    num_epochs=1

    # build go embedding
    logger.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    import os
    logger.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    go_embedding_df = go_embedding.build(
        go_file=go_file,
        seed=config["seed"],
        embedding_dim=config["go_node2vec_embedding_dim"],
        walk_length=config["go_node2vec_walk_length"],
        context_size=config["go_node2vec_context_size"],
        walks_per_node=config["go_node2vec_walks_per_node"],
        p=config["go_node2vec_p"],
        q=config["go_node2vec_q"],
        num_negative_samples=config["go_node2vec_num_negative_samples"],
        save=False,
        num_epochs=num_epochs,
        show_progress=False,
        patience=5,
        early_stopping_delta=0.0002,
        device="cuda"
    )

    # load the graph file
    logger.info(f"Loading data from {iptmnet_graph_file}")
    iptmnet_graph = nx.read_gml(iptmnet_graph_file)

    go_term_type = config["go_term_types"]

    if go_term_type != "none":
        # create features df
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
        embedding_dim=config["iptmnet_node2vec_embedding_dim"],
        walk_length=config["iptmnet_node2vec_walk_length"],
        context_size=config["iptmnet_node2vec_context_size"],
        walks_per_node=config["iptmnet_node2vec_walks_per_node"],
        p=config["iptmnet_node2vec_p"],
        q=config["iptmnet_node2vec_q"],
        num_negative_samples=config["go_node2vec_num_negative_samples"],
        save=False,
        num_epochs=num_epochs,
        patience=5,
        early_stopping_delta=0.0002,
        show_progress=False,
        device="cuda"
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
        min_samples_split=config["rf_min_samples_split"],
        min_samples_leaf=config["rf_min_samples_leaf"],
        criterion=config["rf_criterion"],
        edge_operator=config["rf_edge_operator"],
        plots=False,
        show_progress=False
    )

    logger.info("Clearing memory")
    torch.cuda.empty_cache()
    logger.info("Cleared memory")

    tune.report(au_roc=scores["au_roc"],
                au_pr=scores["au_pr"],
                f1=scores["f1"],
                recall=scores["recall"],
                precision=scores["precision"])