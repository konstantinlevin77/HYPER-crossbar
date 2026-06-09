from datasets import load_dataset 

# Example 1: Load the Protein nodes
proteins = load_dataset("HUBioDataLab/CROssBARv2-KG", data_files="nodes/Protein.csv")

# Example 2: Load Drug-Target Interactions (Edges)
dti = load_dataset("HUBioDataLab/CROssBARv2-KG", data_files="edges/DTI.csv")


print(proteins)
print(proteins["train"].column_names)
print(proteins["train"][0])

print(dti)
print(dti["train"].column_names)
print(dti["train"][0])

# convert to pandas

protein_df = proteins["train"].to_pandas()
dti_df = dti["train"].to_pandas()

print(protein_df.head())
print(dti_df.head())
print(dti_df.columns)
print(dti_df.isna().sum())