import pandas as pd

df = pd.read_csv('data/raw/disgenet.tsv', sep='\t')
df = df[['geneId','diseaseName','score']]
df.to_csv('data/processed/disease_gene.csv', index=False)
