from fastapi import FastAPI
import hvac, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
def get_db_connection():
    vault_token = os.environ.get("VAULT_TOKEN")
    if not vault_token:
        print("VAULT_TOKEN not set in environment.")
        return None

    try:
        client = hvac.Client(url="http://vault:8200", token=vault_token)
        secret = client.secrets.kv.v2.read_secret_version(path="db_creds")
        return secret['data']['data']['connection_string']
    except Exception as e:
        print(f"Vault secret fetch failed: {e}")
        return None


tokenizer = AutoTokenizer.from_pretrained("/outputs/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("/outputs/model")

@app.get("/predict")
def predict(disease: str, drug: str):
    text = disease + " -> " + drug
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    pred = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return {"disease": disease, "drug": drug, "score": pred}
