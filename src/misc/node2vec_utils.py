import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBClassifier
import misc.utils as utils
from p_tqdm import p_umap
import misc.constants as constants
from sklearn.model_selection import RandomizedSearchCV
from misc.torch_model import ANNModel
from skorch import NeuralNetBinaryClassifier
import torch
from skorch.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_curve,roc_curve

# pylint: disable=too-many-arguments
def perform_walks(nodes, graph_original, num_walks, walk_length,p, q,):
    rw = BiasedRandomWalk(graph_original)
    walks = rw.run(nodes, n=num_walks, length=walk_length, p=p, q=q, seed=20)
    return walks

# pylint: disable=too-many-arguments
def build_embedding(graph, name, num_walks, walk_length,window_size, p, q, dimensions, workers, num_iter):
    logger.info("Performing random walks")

    # define a wrapped inner function
    def perform_walks_wrapped(nodes):
        return perform_walks(nodes,graph,num_walks, walk_length, p, q)

    # create chunks
    nodes = graph.nodes()
    node_chunks = list(utils.create_chunks(nodes,100))

    walks_lists = p_umap(perform_walks_wrapped,node_chunks,num_cpus=constants.NUM_OF_JOBS)

    # combine all the list into one big list
    walks = []
    for walk_list in walks_lists:
        walks.extend(walk_list)

    logger.info(f"Number of random walks for '{name}': {len(walks)}")
    logger.info("Training word2vec model")
    
    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
        seed=10
    )

    return model.wv


def build_classifier(dimensions,model_type):
    if model_type == constants.MODEL_TYPE_XGBOOST:
        clf, parameters = create_xgboost_classifier(dimensions)
    elif model_type == constants.MODEL_TYPE_RF:
        clf, parameters = create_random_forest_classifier(dimensions)
    elif model_type == constants.MODEL_TYPE_ANN:
        clf, parameters = create_ann_classifier(dimensions)

    pipeline = Pipeline(steps=[("sc",StandardScaler()),("m", clf)])
    
    grid = RandomizedSearchCV(pipeline,
                            parameters,
                            n_jobs=constants.CV_NUM_OF_JOBS,
                            n_iter=constants.PARAMS["cv_iters"],
                            cv=10,
                            scoring=constants.SCORING,
                            verbose=constants.GRID_VERBOCITY,
                            random_state=10,
                            refit=constants.SCORING,
                )

    return grid

def extract_scaler_and_model(classifier):
    scaler = classifier.best_estimator_.steps[0][1]
    model = classifier.best_estimator_.steps[1][1]
    return (scaler,model)

def create_random_forest_classifier(dimensions):
    clf = RandomForestClassifier(random_state=2)
    parameters = {'m__max_depth':list(range(2,int(dimensions*0.8))),
            "m__criterion":["gini","entropy"],
            'm__max_features':["auto","sqrt", "log2"],
            "m__min_samples_split": list(range(2,100,1)),
            "m__min_samples_leaf": list(range(2,10,1))
    }
    return (clf, parameters)

def create_xgboost_classifier(dimensions):
    clf = XGBClassifier(random_state=2)
    parameters = {'m__max_depth':list(range(2,int(dimensions * 0.8))),
              'm__min_child_weight': list(np.arange(0,1.1,0.1)),
              'm__gamma': list(np.arange(0,15,1)),
              'm__lambda': list(np.arange(0.6,1.1,0.1))
    }
    return (clf,parameters)

def create_ann_classifier(dimensions):
    
    criterion = torch.nn.BCEWithLogitsLoss
    early_stopping = EarlyStopping(monitor="train_loss",patience=20)
    clf = NeuralNetBinaryClassifier(
        module=ANNModel,
        criterion=criterion,
        callbacks=[early_stopping],
        train_split=False,
        verbose=constants.ANN_VERBOCITY,
        device=constants.ANN_DEVICE
    )

    learn_rate = np.arange(0.001,0.1,0.02).tolist()
    momentum = np.arange(0.001,0.1,0.02).tolist()
    epochs = list(range(300,600,100))
    first_layer_units = list(range(10,int(dimensions)))
    second_layer_units = list(range(10,int(dimensions)))
    first_layer_dropout_rate=np.arange(0.1,0.5,0.1).tolist()
    second_layer_dropout_rate=np.arange(0.1,0.5,0.1).tolist()
    parameters = {"m__max_epochs":epochs,
              "m__lr":learn_rate,
              "m__optimizer__momentum":momentum,
              "m__module__input_dim": [dimensions],
              "m__module__first_layer_units":first_layer_units,
              "m__module__second_layer_units":second_layer_units,
              "m__module__first_layer_dropout_rate":first_layer_dropout_rate
    }

    return (clf,parameters)



def train_link_prediction_model(link_embeddings, link_labels, model_type):
    dimensions = np.shape(link_embeddings)[1]
    clf = build_classifier(dimensions=dimensions,model_type=model_type)

    if model_type == constants.MODEL_TYPE_ANN:
        link_labels = link_labels.astype(float)
        clf.fit(link_embeddings, link_labels)
    else:
        clf.fit(link_embeddings, link_labels)
    return clf


def generate_link_embeddings(edges, node_embeddings, binary_operator):
    return [
        generate_link_embedding(src, dst,node_embeddings,binary_operator)
        for src, dst in edges
    ]

def generate_link_embedding(src_node,target_node,node_embeddings,binary_operator):
    src_embedding = node_embeddings[src_node]
    target_embedding = node_embeddings[target_node]
    embedding =  binary_operator(src_embedding,target_embedding)
    return embedding.astype(float)

def evaluate_link_prediction_model(clf,link_embeddings_test, link_labels_test):
    # scale link embeddings
    #link_embeddings_test = scaler.transform(link_embeddings_test)
    if constants.SCORING == constants.SCORING_TYPE_F1:
        return evaluate_f1(clf, link_embeddings_test, link_labels_test)
    elif constants.SCORING == constants.SCORING_TYPE_ROC:
        return evaluate_roc_auc(clf, link_embeddings_test, link_labels_test)


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    #print(clf.predict(link_features))
    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    score = roc_auc_score(link_labels, predicted[:, positive_column])
    curve = roc_curve(link_labels, predicted[:, positive_column])
    return (score, curve)

def evaluate_f1(clf, link_embeddings, link_labels):
    predicted = clf.predict(link_embeddings)
    predicted_proba = clf.predict_proba(link_embeddings)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    score = f1_score(link_labels, predicted)
    curve = precision_recall_curve(link_labels,predicted_proba[:, positive_column])
    return (score, curve)

def plot_roc_auc_curve(clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
    link_features_test = generate_link_embeddings(
        link_examples_test, get_embedding, binary_operator
    )
    result = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return result[1]

def plot_pr_curve(clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
    link_features_test = generate_link_embeddings(
        link_examples_test, get_embedding, binary_operator
    )
    result = evaluate_f1(clf, link_features_test, link_labels_test)
    return result[1]

# define the binary operators
def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0



def train_model(link_embeddings_train, labels_train, link_embeddings_val, labels_val, binary_operator, model_type):
    logger.info(f"Started training for {binary_operator}")
    
    # train
    clf = train_link_prediction_model(link_embeddings_train, labels_train,model_type=model_type)
    
    # evaluate
    score = evaluate_link_prediction_model(
        clf,
        link_embeddings_val,
        labels_val
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score[0],
    }



def get_scaler():
    return StandardScaler()