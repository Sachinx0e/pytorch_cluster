from tqdm import tqdm
import networkx as nx
import pandas as pd
import random
from loguru import logger
import copy

def read_graph():
    # load the iptmnet graph
    iptmnet_graph_file = "data/input/iptmnet_graph_cleaned_directed.gml"
    logger.info(f"Loading data from {iptmnet_graph_file}")
    iptmnet_graph = nx.read_gml(iptmnet_graph_file)
    return iptmnet_graph

def get_negative_edges(nx_graph):
    negative_edges = []
    undirected_grah = nx_graph.to_undirected()
    nodes = undirected_grah.nodes()
    for source_node in tqdm(nodes):
        for target_node in nodes:
            # ensure source node and target node are not the same
            if source_node != target_node:
                # check if a path exists from source node to target node
                path_exists = nx.has_path(undirected_grah,source_node, target_node)
                if path_exists == False:
                    # if no path exists then add the edge
                    negative_edges.append((source_node, target_node))    
                
    return negative_edges


def sample_negative_edges(edges,num_negative=1, seed=10,show_progress=True):
    random.seed(seed)
    positive_edges = edges
    negative_edges = []
    for edge in tqdm(positive_edges,disable=not show_progress):
        
        for _ in range(0,num_negative):
            
            # create negative edge
            negative_edge = _create_corrupted_edge(positive_edges,edge)
                        
            # check if this edge exists already
            while negative_edge in positive_edges:
                negative_edge = _create_corrupted_edge(positive_edges,edge)

            # append to negative edges
            negative_edges.append(negative_edge)

    return negative_edges


def _create_corrupted_edge(positive_edges,edge):
    # radomly pick an egde from positive edges
    random_edge = random.sample(positive_edges,1)[0]

    # get the tail node
    tail_node = random_edge[1]

    # create a new corrupted edge
    negative_edge = (edge[0], tail_node)

    return negative_edge

def get_possible_edges_for_proteins(experimental_proteins,graph):
    all_proteins = list(graph.nodes())

    possible_edges = []
    
    for experimental_protein in experimental_proteins:
        for protein in all_proteins:
            if experimental_protein != protein:
                possible_edges.append((experimental_protein,protein))
                possible_edges.append((protein,experimental_protein))

    return possible_edges

def get_possible_edges_in_graph(graph):
    all_proteins = list(graph.nodes())

    possible_edges = []
    
    for protein_head in all_proteins:
        for protein_tail in all_proteins:
            if protein_head != protein_tail:
                possible_edges.append((protein_head,protein_tail))
                possible_edges.append((protein_tail,protein_head))

    return possible_edges
    

def get_kinases_with_substrates_lt_or_eq_to(nx_graph, num_substrates):
    kinases = []
    nodes = nx_graph.nodes()

    for node in tqdm(nodes):
        # get the num of substrates on this node
        out_degree = nx_graph.out_degree(node)

        if out_degree <= num_substrates:
            kinases.append(node)

    return kinases

def find_common_proteins(nx_graph, nodes):
    """
    Find the nodes that are present in the graph from given list
    """
    kinase_nodes = []

    for node in nx_graph.nodes():
        # get the num of substrates on this node
        out_degree = nx_graph.out_degree(node)
        if out_degree != 0:
            kinase_nodes.append(node)

    nodes_set = set(nodes)
    intersection = nodes_set.intersection(kinase_nodes)
    common_nodes = list(intersection)
    return common_nodes

def get_substrates_for(graph, kinase):
    successors = graph.successors(kinase)
    return list(successors)    

def tag_edges(graph, tag_name, tag_value):
    graph_copy = graph.copy()
    nx.set_edge_attributes(graph_copy,tag_value,tag_name)
    return graph_copy
    pass

def remove_experimental_edges(iptmnet_data, experimental_data, protein_names):        
    # function to map names to index
    def name_to_index(protein):
        protein_index = protein_names.index(protein) + 1
        return protein_index

    # protein names to index
    experimental_data_index_df = pd.DataFrame()
    experimental_data_index_df["kinase_idx"] = experimental_data["KinaseAC"].apply(name_to_index)
    experimental_data_index_df["substrate_idx"] = experimental_data["SubstrateAC"].apply(name_to_index)

    # filter out the indexes that are experimentally validated
    iptmnet_data_copy = iptmnet_data.copy(deep=True)
    for _, row in experimental_data_index_df.iterrows():
        indexes = iptmnet_data_copy[(iptmnet_data_copy["kinase"] == row["kinase_idx"]) & (iptmnet_data_copy["substrate"] == row["substrate_idx"])].index
        iptmnet_data_copy.drop(indexes,inplace=True)

    # map indexes back to protein names
    def index_to_name(index):
        protein_name = protein_names[index - 1]
        return protein_name

    iptmnet_data_copy["kinase"] = iptmnet_data_copy["kinase"].apply(index_to_name)
    iptmnet_data_copy["substrate"] = iptmnet_data_copy["substrate"].apply(index_to_name)
    
    return iptmnet_data_copy


def filter_features_for_proteins(features_df, proteins_to_keep):
    
    # filter
    features_df_filtered = features_df[features_df["protein"].isin(proteins_to_keep)]

    return features_df_filtered

def node_index_to_protein_names(nx_graph, protein_names):
    nodes = nx_graph.nodes
    mapping = {}
    for node in nodes:
        mapping[node] = protein_names[int(node) - 1]
    
    nx_graph = nx.relabel_nodes(nx_graph,mapping)
    return nx_graph

def filter_edges(original_edges, edges_to_remove):
    original_edges_set = set(original_edges)
    edges_to_remove_set = set(edges_to_remove)
    filtered_edges = list(original_edges_set - edges_to_remove_set)
    return filtered_edges




    