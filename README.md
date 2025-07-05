# Domain Suggester GPT-2

This project contains a fine-tuned GPT-2 model that suggests potential domain names based on user-provided business descriptions. The model is trained using Hugging Face Transformers and hosted via a FastAPI backend.

## 🚀 Features

- Fine-tuned GPT-2 model
- FastAPI REST API with `/suggest` endpoint
- Works with custom business-related prompts
- Hosted on Render (or run locally)

## 📦 Repository Structure

```
domain_suggestion/
├── scripts/
│   ├── api.py                 # FastAPI backend for serving suggestions
│   └── dataset_creation.py    # Script to generate training data
├── data/                      # Folder with generated and training data
│   ├── generated_data.json    # Example generated data file
│   └── synthetic_dataset_v1   # Main training data
├── local_model_finetuned/     # Fine-tuned GPT-2 model files
│   ├── config.json
│   ├── merges.txt
│   ├── vocab.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json         # Optional, added if tokenizer is fast-compatible
│   └── pytorch_model.bin      # Main model weights
├── docs/                      # UI and documentation files
│   └── index.html             # Main UI file
├── notebooks/                 # Jupyter notebooks for experiments and training
│   └── gpt2_model_training.ipynb  # Notebook for training the GPT-2 model
└── README.md
```

## 🧠 Model Training

The model is fine-tuned on a custom dataset of domain-style phrases. You can train your own using the `notebooks/gpt2_model_training.ipynb` script.

### Training Notebook

All code for training the model is provided in the Jupyter notebook:

```
notebooks/gpt2_model_training.ipynb
```

Key hyperparameters:
- Epochs: 3
- Batch size: 2
- Max length: 64
- Tokenizer: `GPT2Tokenizer` with `eos_token` as padding

## 🧪 Example Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("local_model_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("local_model_finetuned")
model.eval()

prompt = "Welcome to our smart"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🌐 API Endpoint

### Endpoint: `/suggest`

**Request:**
```json
{
  "prompt": "business description here"
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "domain": "acmallbrand.com",
      "confidence": 0.84
    }
  ]
}
```

## ▶️ Running and Testing the API Locally

You can run the FastAPI backend locally to test domain name suggestions.

### 1. Start the API server

In your terminal, navigate to the project directory and run:

```bash
uvicorn scripts.api:app --reload
```

By default, the API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Test the `/suggest` endpoint

You can test the API using [Swagger UI](http://127.0.0.1:8000/docs) or with a tool like `curl` or Postman.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/suggest" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A platform for smart home automation"}'
```

You should receive a JSON response with domain suggestions.

**Tip:**  
You can also open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser for an interactive API documentation.

### Example Business Descriptions for Testing

- "A platform for smart home automation"
- "Online marketplace for handmade crafts"
- "AI-powered personal finance assistant"
- "Subscription box for healthy snacks"
- "Virtual fitness coaching for remote teams"
- "Eco-friendly cleaning products store"
- "Pet care and grooming booking service"
- "Language learning app"

## 📤 Upload to Hugging Face Hub

```python
from huggingface_hub import login, create_repo, upload_folder

login(token="your_hf_token")
repo_id = "your-username/domain-suggester-gpt2"
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
upload_folder(repo_id=repo_id, folder_path="local_model_finetuned", repo_type="model")
```

## ⚠️ Known Issues

- `tokenizer.json` may be missing if you use `GPT2Tokenizer` instead of `GPT2TokenizerFast`.
- Use `PreTrainedTokenizerFast` for Hugging Face Hub compatibility.
- If Colab download fails due to size, use Google Drive or upload directly to Hugging Face.

## 📄 License

MIT License. This is an open-source academic/experimental project.

## 🤝 Contact

Developed by **Liliia Kryvelova**  
📫 [LinkedIn](https://www.linkedin.com/in/liliiakryvelova/)  
📧 lilia.krivelyova@gmail.com