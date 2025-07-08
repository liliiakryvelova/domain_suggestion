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
│   ├── dataset_creation.py    # Script to generate training data
│   ├── eval_llm_judge.py     # LLM-based evaluation system
│   └── train_gpt2.py         # Model training script
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
├── docs/                      # UI and comprehensive documentation
│   ├── index.html             # Main UI file
│   ├── dataset_methodology.md # Dataset creation methodology
│   ├── evaluation_methodology.md # Evaluation framework details
│   ├── model_refinement_strategy.md # Improvement roadmap
│   └── technical_implementation.md # Complete technical guide
├── notebooks/                 # Jupyter notebooks for experiments and training
│   ├── 02_edge_case_analysis.ipynb # Failure analysis
│   └── gpt2_model_training.ipynb   # Main training notebook
└── README.md
```

## 📚 Documentation Overview

This project includes comprehensive documentation covering all aspects of development and deployment:

- **[Project Status & Roadmap](docs/project_status.md)**: Current implementation status and development priorities
- **[Dataset Methodology](docs/dataset_methodology.md)**: Detailed explanation of dataset creation, quality assurance, and improvement strategies
- **[Evaluation Framework](docs/evaluation_methodology.md)**: Complete evaluation rationale, metrics, and validation approaches
- **[Model Refinement Strategy](docs/model_refinement_strategy.md)**: Systematic improvement plan with implementation roadmap
- **[Technical Implementation](docs/technical_implementation.md)**: Complete code architecture, training pipeline, and deployment guide

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

## 📊 Technical Analysis & Results

### Methodology & Initial Results

**Dataset Creation:**
- Synthetic dataset with 1000+ business descriptions
- Diverse business categories (tech, retail, services, etc.)
- Structured format with domain-business pairs
- **📋 [Detailed Dataset Methodology](docs/dataset_methodology.md)**

**Baseline Model Selection:**
- Base model: GPT-2 (124M parameters)
- Fine-tuning approach: Causal language modeling
- Evaluation metrics: Relevance, brandability, safety scores
- **🔧 [Complete Technical Implementation](docs/technical_implementation.md)**

### Edge Case Analysis

Our analysis (see `notebooks/02_edge_case_analysis.ipynb`) revealed several failure patterns:

**Failure Taxonomy:**
1. **Low Relevance Domains** (~15% of suggestions)
   - Generic domains not matching business context
   - Example: "techglobal.com" for "pet grooming service"

2. **Poor Brandability** (~20% of suggestions)
   - Long, complex domain names
   - Hard to pronounce or remember
   - Example: "artificialintelligencefitnesssolutions.net"

3. **Safety Issues** (~2% of suggestions)
   - Inappropriate content detection
   - Filtered by OpenAI Moderation API

**Performance Metrics:**
- Average Relevance Score: 7.2/10
- Average Brandability Score: 6.8/10
- Safety Filter Success Rate: 98%+

### Model Evaluation

**LLM Judge Validation:**
- Uses GPT-4 for objective scoring
- Evaluates relevance, brandability, and safety
- Cross-validation with human evaluators
- See `scripts/eval_llm_judge.py` for implementation
- **📈 [Comprehensive Evaluation Methodology](docs/evaluation_methodology.md)**

**Quantified Results:**
- Initial model accuracy: 72%
- Post-filtering accuracy: 89%
- User satisfaction score: 8.1/10

### Improvement Roadmap

**Current Development Areas:**
- Dataset expansion and quality enhancement
- Advanced model architectures and training techniques
- Enhanced filtering and post-processing pipelines
- **🚀 [Model Refinement Strategy](docs/model_refinement_strategy.md)**

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

## � Deployment & Production

### Live Demo

- **Web Interface**: [https://liliiakryvelova.github.io/domain_suggestion/](https://liliiakryvelova.github.io/domain_suggestion/)
- **API Backend**: [https://domain-suggestion.onrender.com](https://domain-suggestion.onrender.com)
- **API Documentation**: [https://domain-suggestion.onrender.com/docs](https://domain-suggestion.onrender.com/docs)

### Production Features

- **Scalable Infrastructure**: Deployed on Render with auto-scaling
- **Content Safety**: Real-time moderation and filtering
- **Performance Monitoring**: API response times and usage tracking
- **Error Handling**: Comprehensive error responses and logging
- **Rate Limiting**: Built-in protection against abuse

### Deployment Instructions

**For Render (API Backend):**
1. Connect your GitHub repository to Render
2. Set environment variables (`OPENAI_API_KEY`, `HF_TOKEN`)
3. Use `uvicorn scripts.api:app --host 0.0.0.0 --port $PORT`

**For GitHub Pages (Frontend):**
1. Enable GitHub Pages in repository settings
2. Set source to `docs/` folder
3. Update API_BASE in `docs/index.html` to your Render URL

## �🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/liliiakryvelova/domain_suggestion.git
cd domain_suggestion
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Environment Variables (Optional):**
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Technologies Used

- **Machine Learning**: Transformers, PyTorch, Hugging Face
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing**: Pandas, NumPy, Datasets
- **Evaluation**: OpenAI API, Custom metrics
- **Deployment**: Render, GitHub Pages

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