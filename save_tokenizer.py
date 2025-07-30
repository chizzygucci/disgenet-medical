from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("outputs/tokenizer")
config = AutoConfig.from_pretrained("bert-base-uncased")
config.save_pretrained("outputs/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.save_pretrained("outputs/model")
