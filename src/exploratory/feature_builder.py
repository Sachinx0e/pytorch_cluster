#%%
import pandas as pd

# load data
pro_embeddings = pd.read_csv("../../data/output/pro_embeddings.csv")
pro_embeddings.head()


#%%
rows_df = pro_embeddings[pro_embeddings["protein"] == "PR:Q2M2I8"]
rows_df.iloc[0]
vector = rows_df.drop(columns=["protein"])
vector_np = vector.to_numpy().reshape(-1)
vector_np

# %%
import xmltodict
uniprot_xml = open("../../data/input/uniprot/A6NKT7.xml")
uniprot_xml_string = uniprot_xml.read()

# %%
import re
go_terms = re.findall(r'(?<=GO:)(.*)(?=")',uniprot_xml_string)
go_terms

# %%
import numpy as np
arr1 = np.random.rand(128,)
arr2 = np.random.rand(128,)
arr_m = np.mean([arr1,arr2],axis=0)
print(arr_m.shape)

# %% check nan
import pandas as pd
import numpy as np
features_df = pd.read_csv("../../data/output/features.csv")
features_df = features_df.drop(columns=["protein"])
features_np = features_df.to_numpy()
features_np_sum = np.sum(features_np)
np.isnan(features_np_sum)
# %%
