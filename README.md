# Domain Suggester GPT-2

This project contains a fine-tuned GPT-2 model that suggests potential domain names based on user-provided business descriptions. The model is trained using Hugging Face Transformers and hosted via a FastAPI backend.

## 🚀 Features

- **Fine-tuned GPT-2 model** for domain name generation
- **FastAPI REST API** with comprehensive endpoints
- **Modern Web UI** with responsive design
- **Light/Dark theme toggle** with user preference persistence
- **Content safety filtering** using OpenAI Moderation API
- **Real-time domain validation** and quality scoring
- **Click-to-copy functionality** for generated domains
- **Hosted on Render** (or run locally)

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

## 🌐 API Endpoints

### Endpoint: `/generate-domains`

Generate domain name suggestions based on business description.

**Request:**
```json
{
  "business_description": "AI-powered fitness app",
  "num_domains": 5
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "domain": "aifitness.com",
      "confidence": 0.92
    },
    {
      "domain": "smartworkout.net",
      "confidence": 0.87
    }
  ],
  "status": "success"
}
```

### Endpoint: `/judge-domain`

Analyze and score domain quality for a specific business.

**Request:**
```json
{
  "domain": "aifitness.com",
  "description": "AI-powered fitness app"
}
```

**Response:**
```json
{
  "relevance": 9,
  "brandability": 8,
  "safe": true
}
```

## 🛡️ Content Safety & Security

### Content Moderation

The API includes built-in content safety features to ensure responsible use:

- **OpenAI Moderation API**: All business descriptions are automatically screened for harmful content
- **Real-time Filtering**: Unsafe prompts are blocked before processing
- **Domain Validation**: Generated domains are validated for appropriateness

### Safety Features

1. **Input Validation**: 
   - Blocks offensive, harmful, or inappropriate business descriptions
   - Prevents generation of domains for illegal activities
   - Filters hate speech, harassment, and self-harm content

2. **Output Filtering**:
   - Generated domains are checked for safety
   - Inappropriate domain suggestions are automatically removed
   - Quality scoring includes safety assessments

3. **Rate Limiting**:
   - Domain generation limited to 1-10 suggestions per request
   - Prevents abuse and ensures fair usage

### Unsafe Content Examples

The system automatically blocks requests containing:
- Hate speech or discriminatory language
- Violence or harassment content
- Self-harm or dangerous activities
- Illegal services or products
- Adult content or explicit material
- Scams or fraudulent schemes

### Error Responses

When unsafe content is detected, the API returns:

```json
{
  "status": "blocked",
  "message": "Content flagged by safety filters"
}
```

### Reporting Issues

If you encounter any inappropriate content that wasn't caught by our filters, please report it to: lilia.krivelyova@gmail.com

## ▶️ Running and Testing the API Locally

You can run the FastAPI backend locally to test domain name suggestions.

### 1. Start the API server

In your terminal, navigate to the project directory and run:

```bash
uvicorn scripts.api:app --reload
```

By default, the API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Test the API endpoints

You can test the API using [Swagger UI](http://127.0.0.1:8000/docs) or with a tool like `curl` or Postman.

**Example 1: Generate domain suggestions**

```bash
curl -X POST "http://127.0.0.1:8000/generate-domains" \
     -H "Content-Type: application/json" \
     -d '{"business_description": "AI-powered fitness app", "num_domains": 3}'
```

**Example 2: Judge domain quality**

```bash
curl -X POST "http://127.0.0.1:8000/judge-domain" \
     -H "Content-Type: application/json" \
     -d '{"domain": "aifitness.com", "description": "AI-powered fitness app"}'
```

You should receive JSON responses with domain suggestions or quality scores.

**Tip:**  
You can also open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser for an interactive API documentation.

## 🎨 Web Interface

The project includes a modern, responsive web interface accessible at `docs/index.html`.

### UI Features

- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Light/Dark Mode**: Toggle between themes with preference persistence
- **Real-time Generation**: Live domain suggestions with animated loading
- **Interactive Elements**: Click-to-copy domains with visual feedback
- **Quality Indicators**: Confidence scores and safety badges
- **Modern Styling**: Glass-morphism design with smooth animations

### Accessing the UI

1. **Local Development**: Open `docs/index.html` in your browser
2. **GitHub Pages**: Visit your deployed GitHub Pages URL
3. **With API**: Ensure the API is running for full functionality

### Theme Toggle

The UI includes a persistent light/dark mode toggle:
- **Dark Mode**: Default theme with blue/purple gradients
- **Light Mode**: Clean white theme with subtle accents
- **Auto-save**: User preference is saved to localStorage

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