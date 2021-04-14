from data import data_utils
import networkx as nx
from loguru import logger
import obonet
from tqdm import tqdm
import pandas as pd

def generate_iptmnet_gml():
    # read the data
    _iptmnet_df, graph, _proteins = data_utils.read_data()

    # save the graph to gml
    out_file = "data/input/iptmnet_graph_cleaned_directed.gml"
    nx.write_gml(graph, out_file)
    logger.info(f"Saved graph to : {out_file}")

def generate_iptmnet_gml_only_phosph():
    # read the data
    _iptmnet_df, graph, _proteins = data_utils.read_data(event="p")

    # save the graph to gml
    out_file = "data/input/iptmnet_graph_cleaned_directed_only_phosph.gml"
    nx.write_gml(graph, out_file)
    logger.info(f"Saved graph to : {out_file}")

def generate_go_gml():
    logger.info("Reading go ontology")
    go_graph = obonet.read_obo("data/input/ontology/go.obo")

    # convert to plain old digraph
    logger.info("Generating adj list")
    adj_list = nx.generate_adjlist(go_graph)
    go_graph = nx.parse_adjlist(adj_list)

    # save the graph to gml
    out_file = "data/input/go_graph.gml"
    nx.write_gml(go_graph, out_file)
    logger.info(f"Saved graph to : {out_file}")

def generate_go_parent_mapping():
    # load the obo file
    logger.info("Reading GO ontology file")
    go_graph = obonet.read_obo("data/input/ontology/go.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes by serializing to adjacency matrix
    logger.info("Trimming node attributes")
    edgelist = nx.to_edgelist(go_graph)
    go_graph = nx.DiGraph(edgelist)

    # reverse the direction of edges so that root nodes have highest degree
    go_graph = go_graph.reverse()

    # find the parents of all nodes
    mappings = []
    for node in tqdm(go_graph.nodes()):
        parent = find_root(go_graph,node)
        mapping = (node,parent)
        mappings.append(mapping)

    # save to disk
    df = pd.DataFrame.from_records(mappings,columns=["go_node","root_parent"])
    df.to_csv("data/output/go_parent_mapping.csv",index=False)

def find_root(G,node):
    if list(G.predecessors(node)):  #True if there is a predecessor, False otherwise
        root = find_root(G,list(G.predecessors(node))[0])
    else:
        root = node
    return root