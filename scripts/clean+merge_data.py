import pandas as pd
import numpy as np

# MERGE TABLES S1 & S2:

# Read in both files and transpose/swap row and colums for patient ID and protein ID in Table S2
df_clinical = pd.read_excel('data/raw/cells/Table S1-Patient information and follow-up data.xlsx')
df_protein = pd.read_csv('data/raw/cells/Table S2-Proteins identified by shotgun proteomics analysis.csv', index_col=0)
df_protein_trans = df_protein.T
df_protein_trans.index.name = "Patient_ID"
df_protein_trans = df_protein_trans.reset_index()

# Merge both files
df_final = pd.merge(df_clinical, df_protein_trans, on='Patient_ID', how='inner')
df_final.to_csv('tableS1+S2.csv', index=False)




