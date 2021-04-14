import pandas as pd
from loguru import logger
from tqdm import tqdm
import math
import re
from os import path
import networkx as nx
import misc.utils as utils

def read_data(event=None):
    iptmnet_df = pd.read_csv("data/input/MV_EVENT_human_only.tsv",sep="\t")
    logger.info(f"Loaded data : {iptmnet_df.shape}")

    # extract entities
    substrates  = []
    kinases = []
    graph = nx.DiGraph()
    for _index, row in tqdm(iptmnet_df.iterrows(),total=iptmnet_df.shape[0]):
        event_type = _extract_event_type(row)
        
        # if no event is provided or if event type is equal to provided event type
        if event == None or event == event_type.lower():
            
            # get the entities
            substrate, kinase = extract_entities(row)
            substrates.append(substrate)
            kinases.append(kinase)

            # add new edge to graph
            graph.add_edge(kinase,substrate)

    # combine both the list together
    proteins = []
    proteins.extend(substrates)
    proteins.extend(kinases)

    # remove duplicates
    proteins = list(set(proteins))

    logger.info(f"Total number of unique proteins : {len(proteins)}")
    logger.info(f"Total number of connections: {len(graph.edges)}")
    return iptmnet_df, graph, proteins

def _extract_event_type(row):
    return row["event_label"]

def extract_entities(row):
    substrate = extract_substrate(row)
    kinase = extract_kinase(row)

    return substrate, kinase

def extract_substrate(data_row):
    # check the entity type
    sub_type = data_row["sub_type"]
    
    substrate = ""

    # if entity type is pro id
    if sub_type == "pro_id":
        substrate = data_row["sub_xref"]
        if isinstance(substrate,str) == False and math.isnan(substrate):
            substrate = data_row["SUB_CODE"] 
            
    # if entity type is uniprot ac
    elif sub_type == "uniprot_ac":
        substrate = data_row["SUB_CODE"]

    # else
    else:
        substrate = data_row["sub_form_code"]
    
    substrate_cleaned = clean_entity(substrate)

    # if cleaned entity is all number, then it is a pro_id
    if substrate_cleaned.isnumeric():
        substrate = substrate
    # else keep the cleaned entity
    else:
        substrate = substrate_cleaned

    return substrate

def extract_kinase(data_row):
    # check the entity type
    sub_type = data_row["sub_type"]
    
    kinase = ""

    # if entity type is pro id
    if sub_type == "pro_id":
        kinase = data_row["enz_xref"]
        if isinstance(kinase,str) == False and math.isnan(kinase):
            kinase = data_row["ENZ_CODE"] 
            
    # if entity type is uniprot ac
    elif sub_type == "uniprot_ac":
        kinase = data_row["ENZ_CODE"]

    # else
    else:
        kinase = data_row["enz_form_code"]
    
    kinase_cleaned = clean_entity(kinase)

    # if cleaned entity is all number, then it is a pro_id
    if kinase_cleaned.isnumeric():
        kinase = kinase
    # else keep the cleaned entity
    else:
        kinase = kinase_cleaned

    return kinase

def clean_entity(proteoform):
    proteoform_cleaned = proteoform.replace("PR:","")
    proteoform_cleaned = proteoform_cleaned.split("-")[0]
    
    return proteoform_cleaned

def get_pro_vector(protein_name):
    pass


def read_protein_names():
    proteins_txt = open("data/input/ml_data/iptmnet_network_proteins.txt")
    proteins = proteins_txt.read().splitlines()
    return proteins


def download_from_uniprot():
    protein_names = read_protein_names()
    for _protein in protein_names:
        pass


def get_go_terms(protein_name):

    file_name = f"data/input/uniprot/{protein_name}.xml"

    # check if file exists
    if path.exists(file_name):  
        uniprot_xml = open(file_name).read()
        
        # get all the go terms
        go_terms = re.findall(r'(?<=GO:)(.*)(?=")',uniprot_xml)
        
        # remove duplicates
        go_terms = list(set(go_terms))

        # add the GO: identifier to every term
        go_terms = list(map(lambda x : f"GO:{x}", go_terms))

        return go_terms
    
    # else
    else:

        # return an empty list
        return []

def get_ec_category(protein_name):

    file_name = f"data/input/uniprot/{protein_name}.xml"

    # check if file exists
    if path.exists(file_name):  
        uniprot_xml = open(file_name).read()
        
        # get all the go terms
        ec_terms = re.findall(r'(?<=dbReference type="EC" id=")(.*)(?=")',uniprot_xml)
        
        # remove duplicates
        ec_terms = list(set(ec_terms))

        # keep only the first
        ec_category = "-1"
        if len(ec_terms) > 0:
            ec_term = ec_terms[0]
            ec_category = ec_term.split(".")[0]

        return ec_category
    
    # else
    else:

        # return an empty list
        return []

def get_experimental_edges(experimental_edges_file="data/input/dark_kinase_substrates.txt"):
    # load the experimentally validated relationships
    experimentally_validated = pd.read_csv(experimental_edges_file,sep="\t")
    experimentally_validated = experimentally_validated[["KinaseAC","SubstrateAC"]]

    # clean the experimental data
    kinases = experimentally_validated["KinaseAC"] .apply(utils.clean_proteins).tolist()
    substrates = experimentally_validated["SubstrateAC"] .apply(utils.clean_proteins).tolist()

    # get edge tuples
    experimental_edges = [(kinases[i], substrates[i]) for i in range(0, len(kinases))]

    return experimental_edges

if __name__ == "__main__":
    read_data()


