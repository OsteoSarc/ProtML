import pandas as pd
import numpy as np
import mygene

# Loading Alaa Shotgun Data
print("Loading raw datasets...")
df_protein = pd.read_csv('data/raw/cells/Table S2-Proteins identified by shotgun proteomics analysis.csv', index_col=0)
df_clinical = pd.read_excel('data/raw/cells/Table S1-Patient information and follow-up data.xlsx')


print("Translating UniProt IDs to Gene Symbols...")
mg = mygene.MyGeneInfo()

# Extract the clean UniProt IDs
uniprot_ids = df_protein.index.tolist()

# Query the database
mapping_results = mg.querymany(
    uniprot_ids, 
    scopes='uniprot', 
    fields='symbol', 
    species='human', 
    as_dataframe=True
)

# Clean results and create mapping dictionary
valid_mappings = mapping_results.dropna(subset=['symbol'])
uniprot_to_symbol_dict = valid_mappings['symbol'].to_dict()

# Apply the translation to the dataframe's index
df_protein.index = df_protein.index.map(uniprot_to_symbol_dict)

# Drop rows that failed to translate and average out duplicate Gene Symbols
df_protein = df_protein[df_protein.index.notna()]
df_protein = df_protein.groupby(df_protein.index).mean()


print("Transposing and merging clinical data...")
# Transpose datset so that patients become rows and gene symbols become columns
df_protein_trans = df_protein.T
df_protein_trans.index.name = "Patient_ID"
df_protein_trans = df_protein_trans.reset_index()

# Remove any patients that were excluded from analysis and didn't have a response measurement
# Not to be confused with patients that didn't have response to chemotherapy
df_clinical = df_clinical[
    (df_clinical["Excluded_from_Analysis"] == "No") &
    (df_clinical["Histological_Response_to_Chemotherapy"] != "No_response")
]

# Merge clinical metadata with the translated protein matrix
df_final = pd.merge(df_clinical, df_protein_trans, on='Patient_ID', how='inner')

# Clean text artifacts safely by targeting only string columns
string_columns = df_final.select_dtypes(include=['object']).columns
for col in string_columns:
    df_final[col] = df_final[col].str.replace('â‰¥', '>=', regex=False)

# Save merged dataset as a csv file
output_path = 'data/processed/Alaa_table_merged.csv'
df_final.to_csv(output_path, index=False, na_rep='NULL')
print(f"Saved merged Alaa matrix to {output_path}")

# Map poor responders as 1 and good responders as 0
response_map = {"<90%": 1, ">=90%": 0}
y = df_final["Histological_Response_to_Chemotherapy"].map(response_map)