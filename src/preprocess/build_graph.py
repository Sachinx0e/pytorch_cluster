import pandas as pd
import networkx as nx
from tqdm import tqdm
from loguru import logger


def build_graph(raw_data_file, output_file, graph_type="no_sites",num_rows = 0):
    """
    Build a graph from the raw data

    Parameters
    __________

    raw_data_file : String
                    The location of raw data file

    output_file   : String
                    The location of output file

    graph_type    : String, Optional
                    The type of graph to build. Options are "site_merged", "no_site"
                    Default is "no_sites"
                    "site_merged" will merge substrate and site
                    "no_site" will not merge substrate with sites
    
    num_rows      : int, Optional
                    Specify the number of rows to read to raw data
    
    Returns
    -------
    Networkx.DiGraph
        A directed network X graph.
    """

    # assert params
    assert graph_type in ["site_merged","no_sites"]

    # load the data
    data_file = raw_data_file
    logger.info(f"Loading data from {data_file}")
    raw_data_df = pd.read_csv(f"{data_file}",sep="\t")
    if num_rows != 0:
        raw_data_df = raw_data_df[:num_rows]

    # create a new networkx graph
    graph = nx.DiGraph()

    # loop over the dataframe and add the nodes and edges sequentially
    logger.info("Building graph")
    for index, row in tqdm(raw_data_df.iterrows(),total=len(raw_data_df)):
        # get the data from rows
        enzyme_code = row["enz_form_code"]
        substrate = ""
        if graph_type == "site_merged":
            sub_code = row["SUB_CODE"]
            position = row["POSITION"]
            substrate = f"{sub_code}_{position}"
 
        elif graph_type == "no_sites":
            sub_code = row["sub_form_code"]
            substrate = sub_code

        # add directed from enzyme to substrate
        assert substrate != ""
        graph.add_edge(enzyme_code,substrate)


    # write to disk
    logger.info(f"Writing to {output_file}")
    nx.write_gpickle(graph,f"{output_file}.gpickle")
    nx.write_graphml(graph,f"{output_file}.graphml")
    logger.info("done")
    return graph


if __name__ == "__main__":
    build_graph("data/input/MV_EVENT_human_only.tsv","data/output/iptmnet_knowledge_graph_no_sites")




