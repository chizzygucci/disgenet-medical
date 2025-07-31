#AutoBioML: Automated Drug-Disease Linking via Transformer Models and Secure MLOps



A production-grade MLOps project leveraging FastAPI, HuggingFace, PyTorch, and HashiCorp Vault for secure drug-disease prediction using fine-tuned biomedical models.
Secure secrets management using HashiCorp Vault
FastAPI backend for serving predictions
Dockerized microservices with reproducible environments
Pretrained Transformers for sequence classification
Focus on biomedical NLP pipelines
Git Large File Storage (LFS) to manage large ML models efficiently


#project structure
bio-gen-mlops/
├── outputs/
│ ├── model/ # HuggingFace fine-tuned model
│ │ └── model.safetensors # Stored via Git LFS
│ └── tokenizer/ # Tokenizer directory
├── disgenet.py # Data extraction logic
├── mergectd.py # Merging external datasets (e.g., CTD)
├── modeltrain.py # Model training script
├── inferenceserve.py # FastAPI prediction API
├── save_tokenizer.py # Script to save tokenizer locally
├── dockerfile # Container image for the API
├── docker-compose.yaml # Orchestrates services (API + Vault)
├── requirements.txt # Python dependencies
└── .gitattributes # Git LFS tracking configuration



##  API Endpoint

**GET** `/predict`

### Query Parameters:
- `disease` – Disease name (e.g., `"Diabetes"`)
- 'drug` – Drug name (e.g., `"Metformin"`)

### Example Call:

curl "http://localhost:8000/predict?disease=Diabetes&drug=Metformin"


#Example response
{
  "disease": "Diabetes",
  "drug": "Metformin",
  "score": 0.853
}



## Vault Integration (Optional in This Demo)

This project includes code for integrating [HashiCorp Vault](https://www.vaultproject.io/) to securely retrieve secrets like database connection strings:

Vault Integration (Optional in This Demo)
This project includes support for integrating HashiCorp Vault to securely fetch secrets (e.g., a database connection string). While no actual database is used in this version, 
the code anticipates production-ready practices by securely attempting to retrieve credentials from Vault if a VAULT_TOKEN is provided.

The function below demonstrates how the application securely attempts to retrieve a database connection string from HashiCorp Vault.

If `VAULT_TOKEN` is not set in the environment, Vault is gracefully bypassed and `None` is returned. This makes the app safe for both local/demo use and production environments.

```python
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


# If no token is found or Vault is not running, the function exits silently with a helpful message, allowing the app to run without a database.


## ⚠ Challenges & Solutions

###  Docker Compose Error: KeyError: 'ContainerConfig'

**Problem:**
while running docker-compose up --build
I encounter the following error:
File "/usr/lib/python3/dist-packages/compose/service.py", line 1579, in get_container_data_volumes
    container.image_config['ContainerConfig'].get('Volumes') or {}
KeyError: 'ContainerConfig'


**Cause:**

This error typically occurs when:
•	Docker Compose attempts to read metadata from a corrupted or incomplete image.
•	There are leftover containers or volumes from a previous failed build.
•	An invalid or outdated base image is used in the Dockerfile.


**Solution:**
1.	Stop and clean up any old or orphaned containers/volumes:

docker-compose down --volumes --remove-orphans
2.	Prune unused Docker resources (⚠removes unused images, containers, volumes):

docker system prune -af --volumes
3.	Rebuild and start the containers fresh:

docker-compose build --no-cache
docker-compose up --force-recreate


**Problem:**


NumPy Compatibility Issue (PyTorch & Transformers)
During setup, I encounterd errors related to safetensors, torch, or transformers. These are often caused by incompatibility with NumPy version 2.x.

 #cause
Many machine learning libraries (including PyTorch, Transformers, and JAX) were built against NumPy 1.x APIs. NumPy 2.x introduces breaking changes that are not yet fully supported by all these libraries.

ImportError: numpy.core.multiarray failed to import

#solution
To avoid these issues, pin NumPy to a version below 2.0.
In my requirements.txt: i impute "numpy<2.0"


This ensures compatibility with existing ML libraries and avoids runtime crashes during Docker build or container inference.



