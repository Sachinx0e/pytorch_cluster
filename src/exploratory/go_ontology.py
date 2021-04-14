# %%
import obonet
from loguru import logger
logger.info("Reading go ontology")
go_graph = obonet.read_obo("../../data/input/ontology/go.obo")
logger.info("Done reading go ontology")

# %%
# show the number of nodes
logger.info(f"Number of nodes : {len(go_graph.nodes())}")
logger.info(f"Number of edges : {len(go_graph.edges())}")
nodes = list(go_graph.nodes())
# %%
import networkx as nx
logger.info("Generating adjacency matrix")
adj_mat = nx.to_numpy_matrix(go_graph)
logger.info("Done")
logger.info(f"Matrix shape : {adj_mat.shape}")
# %%

# %%
