from tqdm import tqdm
from loguru import logger
import numpy as np
import pandas as pd
import data.data_utils as du
import misc.graph_utils as gu
from functools import reduce
import misc.utils as utils
from p_tqdm import p_map
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing


"""
This method is used to build a feature matrix to be used as a part of GNN layer
"""
def build_features(graph):
    
    if graph == None:
      graph = gu.read_graph()

    proteins = graph.nodes()

    # load pro embeddings
    logger.info("Loading pro embeddings")
    pro_embeddings_df = pd.read_csv("data/output/pro_embeddings.csv")

    # load go embeddings
    logger.info("Loading go embeddings")
    go_embeddings_df = pd.read_csv("data/output/go_embeddings.csv")

    # load sequence embeddings
    logger.info("Loading sequence embedddings")
    seq_embeddings_df = pd.read_csv("data/output/seq_embeddings.csv")

    # create empty array to hold the data
    pro_vector = pro_embeddings_df.shape[1] - 1
    go_vector = go_embeddings_df.shape[1] - 1
    seq_vector = seq_embeddings_df.shape[1] - 1
    num_rows = len(proteins)
    num_columns = pro_vector + go_vector + seq_vector
    features_arr = np.empty(shape=(num_rows, num_columns))

    for index, protein_name in tqdm(enumerate(proteins), total=len(proteins)):
        
        # pro 
        pro_vector = get_pro_vector(protein_name,pro_embeddings_df)

        # seq
        seq_vector = get_seq_vector(protein_name, seq_embeddings_df)
                
        # go vector
        go_vector = build_go_vector(protein_name,go_embeddings_df)

        # feature vector
        feature_vector = np.concatenate([pro_vector, go_vector, seq_vector])

        # append to features array
        features_arr[index] = feature_vector
    
    features_pd = pd.DataFrame(data=features_arr)
    features_pd.insert(0,"protein",proteins)

    features_pd.to_csv("data/output/features.csv",index=False)

    return features_pd

def build_features_only_go(graph=None,go_embeddings_df=None,save=True):
    
    if graph == None:
      graph = gu.read_graph()

    proteins = graph.nodes()

    # load go embeddings
    if go_embeddings_df is None:
      logger.info("Loading go embeddings")
      go_embeddings_df = pd.read_csv("data/output/go_embeddings_copy.csv")

    # create empty array to hold the data
    go_vector = go_embeddings_df.shape[1] - 1
    num_rows = len(proteins)
    num_columns = go_vector
    features_arr = np.empty(shape=(num_rows, num_columns))

    for index, protein_name in tqdm(enumerate(proteins), total=len(proteins)):
                
        # go vector
        go_vector = build_go_vector(protein_name,go_embeddings_df)

        # append to features array
        features_arr[index] = go_vector
    
    features_pd = pd.DataFrame(data=features_arr)
    features_pd.insert(0,"protein",proteins)
    
    if save == True:
      features_pd.to_csv("data/output/features_only_go.csv",index=False)

    return features_pd

def build_features_only_go_term_type(term_type,
                                     graph=None,
                                     go_embeddings_df=None,
                                     parent_mapping_file="data/output/go_parent_mapping.csv",
                                     binary_operator="hadamard",
                                     save=True
                                     ):
    
    if graph == None:
      graph = gu.read_graph()

    proteins = graph.nodes()

    # load go embeddings
    if go_embeddings_df is None:
      logger.info("Loading go embeddings")
      go_embeddings_df = pd.read_csv("data/output/go_embeddings_copy.csv")

    # load parent mapping
    logger.info(f"Loading go parent mappings - {parent_mapping_file}")
    parent_mapping_df = pd.read_csv(parent_mapping_file)

    # create empty array to hold the data
    go_vector = go_embeddings_df.shape[1] - 1
    num_rows = len(proteins)
    num_columns = go_vector
    features_arr = np.empty(shape=(num_rows, num_columns))

    # define work
    def do_work(protein_name):
      go_vector = build_go_vector_for_term_type(protein_name,go_embeddings_df,parent_mapping_df,term_type,binary_operator=binary_operator)
      go_vector = np.reshape(go_vector,(-1,num_columns))
      return go_vector

    # define pool
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count-1)
    
    # do work
    logger.info("Getting go vector for term")
    features_np_list = pool.map(do_work, proteins)

    features_arr = np.concatenate(features_np_list,axis=0)

    features_pd = pd.DataFrame(data=features_arr)
    features_pd.insert(0,"protein",proteins)

    if save == True:
      features_pd.to_csv(f"data/output/features_only_go_{term_type}.csv",index=False)

    return features_pd

def build_ec_category():
  proteins = du.read_protein_names()

  ec_mapping = []
  for protein in tqdm(proteins):
    ec_code = du.get_ec_category(protein)
    ec_mapping.append((protein,ec_code))
  
  ec_df = pd.DataFrame.from_records(ec_mapping, columns=["protein","ec_category"])
  ec_df.to_csv("data/output/ec_category.csv",index=False)


def get_pro_vector(protein_name, embedding_df):
  rows_df = embedding_df[embedding_df["protein"] == f"PR:{protein_name}"]
  
  # if protein found
  if len(rows_df.index) > 0:
    rows_df = rows_df.drop(columns=["protein"])
    row_vector = rows_df.iloc[0].to_numpy().reshape(-1)
    return row_vector

  # if protein not found
  else:
    # create an empty vector
    num_dimensions = embedding_df.shape[1]-1
    empty_vector = np.zeros(shape=(num_dimensions,))
    return empty_vector

def get_seq_vector(protein_name, embedding_df):
  rows_df = embedding_df[embedding_df["protein"] == protein_name]
  
  # if protein found
  if len(rows_df.index) > 0:
    rows_df = rows_df.drop(columns=["protein"])
    row_vector = rows_df.iloc[0].to_numpy().reshape(-1)
    return row_vector

  # if protein not found
  else:
    # create an empty vector
    num_dimensions = embedding_df.shape[1]-1
    empty_vector = np.zeros(shape=(num_dimensions,))
    return empty_vector


def build_go_vector(protein_name, go_embeddings_df):
  # get go terms
  go_terms = du.get_go_terms(protein_name)

  # get embeddings for these go terms
  num_rows = len(go_terms)
  num_columns = go_embeddings_df.shape[1] - 1
  go_vectors = np.zeros(shape=(num_rows,num_columns))
  for index, go_term in enumerate(go_terms):
    go_vector = _get_go_vector(go_term,go_embeddings_df)
    go_vectors[index] = go_vector

  # check if all the elements in go_vectors are not zero
  if np.all(go_vectors == 0) == False:
    # calculate a sum of all the vectors
    go_vectors_mean = np.sum(go_vectors,axis=0)
    return go_vectors_mean
  else:
    return np.zeros(shape=(num_columns,))

def build_go_vector_for_term_type(protein_name, go_embeddings_df,parent_mapping_df,term_type,binary_operator="hadamard"):

  # get go terms
  go_terms = du.get_go_terms(protein_name)

  if term_type == "bp":
    go_terms = filter_children_for_parent(go_terms,"GO:0008150",parent_mapping_df)
  elif term_type == "cc":
    go_terms = filter_children_for_parent(go_terms,"GO:0005575",parent_mapping_df)
  elif term_type == "mp":
    go_terms = filter_children_for_parent(go_terms,"GO:0003674",parent_mapping_df)
  elif term_type == "all":
    go_terms = go_terms
  else:
    raise Exception(f"Unrecongnized term type : {term_type}")

  # get embeddings for these go terms
  num_rows = len(go_terms)
  num_columns = go_embeddings_df.shape[1] - 1
  go_vectors = np.zeros(shape=(num_rows,num_columns))
  for index, go_term in enumerate(go_terms):
    go_vector = _get_go_vector(go_term,go_embeddings_df)
    go_vectors[index] = go_vector

  # check if all the elements in go_vectors are not zero
  if np.all(go_vectors == 0) == False:
    if binary_operator == "l1":
      return reduce(utils.operator_l1,go_vectors)
    elif binary_operator == "l2":
      return reduce(utils.operator_l2,go_vectors)
    elif binary_operator == "hadamard":
      return reduce(utils.operator_hadamard,go_vectors)
    elif binary_operator == "avg":
      return reduce(utils.operator_avg,go_vectors)
    elif binary_operator == "cat":
      return reduce(utils.operator_cat,go_vectors)
  else:
    return np.zeros(shape=(num_columns,))
  

def filter_children_for_parent(go_terms, parent_go_term, mappings_df):
  child_go_terms = []
  for term in go_terms:
    mappings_df_filtered = mappings_df[mappings_df["go_node"] == term]
    if mappings_df_filtered.shape[0] > 0:
      parent = mappings_df_filtered.iloc[0]["root_parent"]
      if parent == parent_go_term:
        child_go_terms.append(term)

  return child_go_terms

def _get_go_vector(go_term, go_embeddings_df):
  rows_df = go_embeddings_df[go_embeddings_df["protein"] == go_term]
  
  # if term found
  if len(rows_df.index) > 0:
    rows_df = rows_df.drop(columns=["protein"])
    row_vector = rows_df.iloc[0].to_numpy().reshape(-1)
    return row_vector

  # if term not found
  else:
    # create an empty vector
    num_dimensions = go_embeddings_df.shape[1]-1
    empty_vector = np.zeros(shape=(num_dimensions,))
    return empty_vector

if __name__ == "__main__":
  features_df = build_features(None)
  