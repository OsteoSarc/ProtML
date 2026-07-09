import pandas as pd
import numpy as np
import mygene
from sklearn.preprocessing import StandardScaler

# STEP 1: LOAD RAW DATA
print("Loading raw datasets...")

# Alaa Shotgun Data
df_protein = pd.read_csv('data/raw/cells/Table S2-Proteins identified by shotgun proteomics analysis.csv', index_col=0)
df_clinical = pd.read_excel('data/raw/cells/Table S1-Patient information and follow-up data.xlsx')

# Zhang DIA Data
df_zhang = pd.read_excel('data/raw/Zhang_table 1.xlsx', index_col=0)

# STEP 2: TRANSLATE UNIPROT TO GENE SYMBOLS
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

# STEP 3: TRANSPOSE & MERGE CLINICAL DATA
print("Transposing and merging clinical data...")
# Patients become rows, Gene Symbols become columns
df_protein_trans = df_protein.T
df_protein_trans.index.name = "Patient_ID"
df_protein_trans = df_protein_trans.reset_index()

# Merge clinical metadata with the translated protein matrix
df_final = pd.merge(df_clinical, df_protein_trans, on='Patient_ID', how='inner')

# Clean text artifacts safely by targeting only string columns
string_columns = df_final.select_dtypes(include=['object']).columns
for col in string_columns:
    df_final[col] = df_final[col].str.replace('Ã¢â€°Â¥', '>=', regex=False)
    df_final[col] = df_final[col].str.replace('â‰¥', '>=', regex=False)

output_path = 'data/processed/tableS1+S2.csv'
df_final.to_csv(output_path, index=False, na_rep='NULL')
print(f"Saved merged Alaa matrix to {output_path}")

# STEP 4: FEATURE INTERSECTION & SCALING
print("Intersecting and scaling features for Machine Learning...")

# Find proteins detected by both DIA (Zhang) and Shotgun (Alaa)
shared_genes = df_zhang.index.intersection(df_protein.index)

# Verify critical resistance hubs made it through
if 'MAEA' in shared_genes and 'MRPL4' in shared_genes:
    print("Success: MAEA and MRPL4 survived the intersection and are ready for modeling.")
else:
    print("Warning: MAEA or MRPL4 did not survive the intersection. Check your Gene Symbol mappings.")

# Slice both datasets down to only the shared genes
# Transpose (.T) so Patients are rows and Genes are columns
X_train_raw = df_zhang.loc[shared_genes].T
X_test_raw = df_protein.loc[shared_genes].T

# Initialize independent scalers to prevent machine batch effects
scaler_train = StandardScaler()
scaler_test = StandardScaler()

# Fit and transform the discovery set (Zhang)
X_train_final = pd.DataFrame(
    scaler_train.fit_transform(X_train_raw), 
    index=X_train_raw.index, 
    columns=X_train_raw.columns
)

# Fit and transform the validation set (Alaa)
X_test_final = pd.DataFrame(
    scaler_test.fit_transform(X_test_raw), 
    index=X_test_raw.index, 
    columns=X_test_raw.columns
)

# Export the final Machine Learning matrices
train_out_path = 'data/processed/X_train_zhang.csv'
test_out_path = 'data/processed/X_test_alaa.csv'

X_train_final.to_csv(train_out_path)
X_test_final.to_csv(test_out_path)

print("\n--- Pipeline Complete ---")
print(f"Final Training Matrix Shape (Zhang): {X_train_final.shape}")
print(f"Saved Training Matrix to: {train_out_path}")
print(f"Final Testing Matrix Shape (Alaa): {X_test_final.shape}")
print(f"Saved Testing Matrix to: {test_out_path}")