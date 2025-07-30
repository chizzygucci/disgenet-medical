import pandas as pd

df_ctd = pd.read_csv('data/raw/ctd.tsv', sep='\t')
df_ctd = df_ctd[['ChemicalName','DiseaseName','AssociationScore']]
df_ctd.to_csv('data/processed/drug_disease.csv', index=False)

# Merge disease-gene + drug-disease
gene = pd.read_csv('data/processed/disease_gene.csv')
drug = pd.read_csv('data/processed/drug_disease.csv')
merged = pd.merge(drug, gene, left_on='DiseaseName', right_on='diseaseName')
merged.to_csv('data/processed/drug_gene_disease.csv', index=False)
