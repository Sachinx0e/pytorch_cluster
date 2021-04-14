from data import data_utils
import networkx as nx
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import obonet

def generate_stats_iptmnet():
    # read the data
    _iptmnet_df, graph, _proteins = data_utils.read_data()

    # plot the out degree distribution
    degree_tuples = list(graph.out_degree)
    
    # create dataframe
    degree_df = pd.DataFrame.from_records(degree_tuples,columns=["kinase","out_degree"])
    degree_df = degree_df.sort_values(by=["out_degree"],ascending=False)

    # create a bar plot
    fig = px.line(degree_df[:100], x="kinase", y="out_degree", title="Kinases by degree")
    fig.update_traces(marker_color='green')
    fig.write_image("results/exploratory/figures/out_degree.png")
    
    # plot the in degree distribution
    degree_tuples = list(graph.in_degree)
    
    # create dataframe
    degree_df = pd.DataFrame.from_records(degree_tuples,columns=["substrate","in_degree"])
    degree_df = degree_df.sort_values(by=["in_degree"],ascending=False)

    # create a bar plot
    fig = px.line(degree_df[:100], x="substrate", y="in_degree",title="Substrates by degree")
    fig.update_traces(marker_color='green')
    fig.write_image("results/exploratory/figures/in_degree.png")

def generate_stats_go():
    logger.info("Reading go ontology")
    go_graph = obonet.read_obo("data/input/ontology/go.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes by serializing to adjacency matrix
    logger.info("Trimming node attributes")
    edgelist = nx.to_edgelist(go_graph)
    go_graph = nx.DiGraph(edgelist)

    # reverse the direction of edges
    go_graph = go_graph.reverse()
 
    # get the out degree
    degree_tuples = list(go_graph.out_degree)
    degree_df = pd.DataFrame.from_records(degree_tuples,columns=["go_term","degree"])
    degree_df = degree_df.sort_values(by=["degree"],ascending=False)

    # print top 10 go terms
    print(degree_df.head())

    # print degree of GO
    print(list(go_graph.predecessors("GO:0110165")))

    



