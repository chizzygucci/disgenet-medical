import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/drug_gene_disease.csv')
df['text'] = df['DiseaseName'] + " -> " + df['ChemicalName']
df['label'] = (df['AssociationScore'] > df['score']).astype(int)
train, val = train_test_split(df, test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
train_enc = train.apply(lambda row: {'input_ids': tokenizer(row['text'], truncation=True, padding='max_length')['input_ids'], 'label': row['label']}, axis=1)
# Simplified dataset
from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return {'input_ids': self.df.iloc[idx]['input_ids'], 'labels': self.df.iloc[idx]['label']}
train_ds = SimpleDataset(train_enc.tolist())
val_ds = SimpleDataset(val.apply(lambda r: {'input_ids': tokenizer(r['text'], truncation=True, padding='max_length')['input_ids'], 'label': r['label']}, axis=1).tolist())

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
trainer = Trainer(model=model,
                  args=TrainingArguments(output_dir="outputs", evaluation_strategy="epoch", per_device_train_batch_size=16, num_train_epochs=3),
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
model.save_pretrained("outputs/model")
tokenizer.save_pretrained("outputs/tokenizer")
