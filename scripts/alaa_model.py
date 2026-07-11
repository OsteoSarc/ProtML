import pandas as pd
import numpy as np
import mygene
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

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

# Pull list of genes (column names of transposed protein data) and convert expression to float
gene_cols = [c for c in df_protein_trans.columns if c != "Patient_ID"]
X = df_final[gene_cols].astype(float)

# Create a filter to remove genes missing in too many patients (threshold of 0.25)
missing_frac = X.isna().mean(axis=0)
X = X.loc[:, missing_frac <= 0.25]
print(f"Genes remaining after absence of data filter: {X.shape[1]}")

X_arr = X.values
y_arr = y.values

# Calculate Mann-Whitney score to rank each gene by how differently it behaves between good vs poor responders
def mannwhitney_score(X_mat, y_vec):
    group0 = y_vec == 0
    group1 = y_vec == 1
    _, p_values = stats.mannwhitneyu(X_mat[group0], X_mat[group1], axis=0, alternative="two-sided")
    return -p_values

top_n_genes = 100

# Pipeline to group preprocessing and model so each step refits per fold and avoids leakage
pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")), # Fill missing values with each gene's median
    ("variance_filter", VarianceThreshold(threshold=0.0)), # Drop genes with zero variance (no info)
    ("univariate_select", SelectKBest(score_func = mannwhitney_score, k = top_n_genes)), # Keep top 100 most different genes in group
    ("scale", StandardScaler()), # Standardize genes to mean 0 and std 1
    ("classify", LogisticRegressionCV( # L1-regularized logistic regression to handle class imbalance
        Cs = 10, 
        cv = 5, 
        penalty = "l1", 
        solver = "liblinear",
        class_weight = "balanced", 
        max_iter = 5000, 
        random_state = 0
        )
    )
])

loo = LeaveOneOut()

print("Running Pipeline...")
# Run LOOCV twice: once for hard 0/1 predictions and once for probabilities (needed for AUC)
y_pred = cross_val_predict(pipeline, X_arr, y_arr, cv=loo, method="predict")
y_proba = cross_val_predict(pipeline, X_arr, y_arr, cv=loo, method="predict_proba")[:, 1]