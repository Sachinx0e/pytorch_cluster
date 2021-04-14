# %%
import obonet
from loguru import logger
logger.info("Reading pro ontology")
pro_graph = obonet.read_obo("../../data/input/ontology/pro_reasoned.obo")
logger.info("Done reading pro ontology")


# %%
# show the number of nodes
logger.info(f"Number of nodes : {len(pro_graph.nodes())}")
logger.info(f"Number of edges : {len(pro_graph.edges())}")
nodes = list(pro_graph.nodes())
