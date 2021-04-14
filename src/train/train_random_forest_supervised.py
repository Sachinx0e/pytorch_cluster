from loguru import logger
import networkx as nx
import pandas as pd
import torch_geometric as geometric
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import misc.graph_utils as gu
import numpy as np
from tqdm import tqdm
import data.feature_builder as feature_builder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from misc import utils
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score,make_scorer, recall_score, precision_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from visualization import plotter
import data.data_utils as du
from p_tqdm import p_map
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

def train(
          max_depth,
          criterion,
          max_features,
          min_samples_split,
          min_samples_leaf,  
          iptmnet_graph=None,
          experimental_edges=None,
          iptmnet_embedding_df=None,
          features_df=None,
          random_state=20,
          edge_operator="hadamard",
          plots=True,
          show_progress=False,
          save=False,
          predict=False,
          idg_predictions_file="results/predictions/idg_predictions_test.csv",
          iptmnet_predictions_file="results/predictions/iptmnet_understudied_predictions_test.csv" 
        ):

    # load the iptmnet graph
    if iptmnet_graph == None:
        iptmnet_graph_file = "data/input/iptmnet_graph_cleaned_directed.gml"
        logger.info(f"Loading data from {iptmnet_graph_file}")
        iptmnet_graph = nx.read_gml(iptmnet_graph_file)

    # positive edges
    positive_edges = list(iptmnet_graph.edges())
    
    # filter experimental edges
    if experimental_edges is None:
        experimental_edges = du.get_experimental_edges()
    logger.info(f"Before filtering : {len(positive_edges)}")
    positive_edges = gu.filter_edges(positive_edges,experimental_edges)
    logger.info(f"After filtering : {len(positive_edges)}")

    # split into train and test
    positive_edges_train, positive_edges_test = train_test_split(positive_edges,random_state=20,shuffle=True)

    # sample negative edges for both train and test sets
    negative_edges_train = gu.sample_negative_edges(positive_edges_train,seed=20,show_progress=show_progress)
    negative_edges_test = gu.sample_negative_edges(positive_edges_test,seed=20,show_progress=show_progress)

    # create labels
    positive_train_labels = np.ones((len(positive_edges_train),))
    positive_test_labels = np.ones((len(positive_edges_test),))
    negative_train_labels = np.zeros((len(negative_edges_train),))
    negative_test_labels = np.zeros((len(negative_edges_test),))

    # combine both positive and negative
    train_edges = positive_edges_train + negative_edges_train
    test_edges = positive_edges_test + negative_edges_test

    # combine both positive and negative labels
    train_labels = np.concatenate([positive_train_labels, negative_train_labels],axis=0)
    test_labels = np.concatenate([positive_test_labels, negative_test_labels],axis=0)

    # shuffle them
    train_edges, train_labels = shuffle(train_edges,train_labels,random_state=20)
    test_edges, test_labels = shuffle(test_edges,test_labels,random_state=20)
    
    # load the iptmnet embeddings
    if iptmnet_embedding_df is None:
        iptmnet_embeddings_file = "data/output/iptmnet_embeddings.csv"
        logger.info(f"Loading iptmnet embeddings - {iptmnet_embeddings_file}")
        iptmnet_embedding_df = pd.read_csv(iptmnet_embeddings_file)

    # load all features
    if features_df is None:
        features_file = "data/output/features.csv"
        logger.info(f"Loading features - {features_file}")
        features_df = pd.read_csv(features_file)

    # concatenate embeddings
    embeddings_pd = pd.merge(left=iptmnet_embedding_df,right=features_df,left_on="protein",right_on="protein")

    # get edge embeddings
    logger.info("Getting edge embeddings")
    train_edge_embeddings = utils.get_embeddings_vector(train_edges,embeddings_pd,operator=edge_operator,show_progress=show_progress)
    test_edge_embeddings = utils.get_embeddings_vector(test_edges,embeddings_pd,operator=edge_operator,show_progress=show_progress)

    # define the model
    rf_model = RandomForestClassifier(random_state=20)
    
    # define params
    parameters = {'m__max_depth':[int(np.shape(train_edge_embeddings)[1] * max_depth)],
                "m__criterion":[criterion],
                'm__max_features':[max_features],
                "m__min_samples_split": [min_samples_split],
                "m__min_samples_leaf": [min_samples_leaf]
                }
        
    # define the pipeline
    cv = StratifiedKFold(n_splits=10)
    scaler = StandardScaler() 
    steps = [('s',scaler),("m",rf_model)]
    pipeline = Pipeline(steps=steps)

    # define scorer
    scorer = make_scorer(f1_score,average="binary",pos_label=1)

    # fit
    logger.info("Started training")
    grid = GridSearchCV(pipeline,parameters, n_jobs=16,cv=cv, scoring=scorer, verbose=1, refit=True)
    grid.fit(X=train_edge_embeddings, y=train_labels)

    # make prediction
    logger.info("Making predictions")
    labels_predicted = grid.predict(test_edge_embeddings)
    labels_predicted_proba = grid.predict_proba(test_edge_embeddings)

    # calculate scores
    scores = {
        "f1" : f1_score(test_labels,labels_predicted,average="binary",pos_label=1),
        "recall" : recall_score(test_labels,labels_predicted,average="binary",pos_label=1),
        "precision" : precision_score(test_labels,labels_predicted,average="binary",pos_label=1),
        "au_roc": roc_auc_score(test_labels,labels_predicted_proba[:,1]),
        "au_pr": average_precision_score(test_labels,labels_predicted_proba[:,1])
    }

    logger.info(scores)

    if plots == True:
        # calculate PR baseline
        pr_baseline = len(test_labels[test_labels==1]) / len(test_labels)
        
        # calculate the curves
        curves = {
            "pr_curve" : precision_recall_curve(test_labels,labels_predicted_proba[:,1],pos_label=1),
            "roc_curve" : roc_curve(test_labels,labels_predicted_proba[:,1],pos_label=1),
        }

        # plot pr curve
        fig_pr, ax_pr = plt.subplots()
        plotter.plot_pr_curve(fig_pr,ax_pr,curves["pr_curve"],"PR curve")
        plotter.plot_pr_curve_baseline(fig_pr,ax_pr,pr_baseline)
        fig_pr.savefig("results/figures/pr_curve_embeddings.png")

        # plot roc curve
        fig_roc, ax_roc = plt.subplots()
        plotter.plot_roc_curve(fig_roc,ax_roc,curves["roc_curve"],"ROC curve")
        plotter.plot_roc_curve_baseline(fig_roc,ax_roc)
        fig_roc.savefig("results/figures/roc_curve_embeddings.png")

    # if predict, then predict possible interactions for experimental proteins
    if predict == True:
        
        # predict idg
        predictions_pd = _predict(grid,iptmnet_graph,embeddings_pd,edge_operator,show_progress,"idg")
        predictions_pd.to_csv(idg_predictions_file,index=False)
        logger.info(f"Predictions saved to {idg_predictions_file}")              

    return scores

def _predict(grid,
             iptmnet_graph,
             embeddings_pd,
             edge_operator,
             show_progress,
             set_type,
             understudied_threshold=2   
            ):
    
    logger.info(f"##### PREDICTING FOR {set_type} PROTEINS ######")

    predict_proteins = []
    if set_type == "idg":
        # load the idg proteins list
        idg_proteins = pd.read_csv("data/input/idg_protein_list.csv")["Substrates"].tolist()
        predict_proteins.extend(idg_proteins)

    elif set_type == "iptmnet":
        # get the understudied proteins from iptmnet
        node_degrees = iptmnet_graph.degree(iptmnet_graph.nodes())
        for node_degree in node_degrees:
            degree = node_degree[1]
            if degree <= understudied_threshold:
                predict_proteins.append(node_degree[0])
    
    else:
        raise Exception(f"Unknown set_type : {set_type}")

    # generate permutations for all possible edges
    logger.info(f"Number of predict proteins for {set_type} set: {len(predict_proteins)}")
    logger.info("Getting all possible edges for IDG proteins")
    all_possible_edges = gu.get_possible_edges_for_proteins(predict_proteins,iptmnet_graph)
    logger.info(f"Number of possible edges: {len(all_possible_edges)}")

    # get embeddings for possible edges
    logger.info("Getting edge embeddigs for possible edges")
    possible_edge_embeddings = utils.get_embeddings_vector(all_possible_edges,embeddings_pd,operator=edge_operator,show_progress=show_progress)

    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count-1)

    # predict 
    logger.info("Predicting")
    predictions = pool.map(lambda link_embedding : grid.predict_proba(np.reshape(link_embedding,(1,-1)))[0][1],possible_edge_embeddings)
    
    # merge edges and predictions
    logger.info("Building prediction tuples")
    prediction_tuples = p_map(lambda index: _build_prediction_tuple(all_possible_edges[index],predictions[index]),range(0,len(all_possible_edges)))

    # transform to dataframe
    logger.info("Transforming to dataframe")
    predictions_pd = pd.DataFrame(prediction_tuples,columns=["source","target","prediction"])
    
    return predictions_pd


def _build_prediction_tuple(possible_edge,prediction):
    return (possible_edge[0],possible_edge[1],prediction)
