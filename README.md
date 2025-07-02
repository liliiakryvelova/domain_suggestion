# Domain Suggestion LLM Project

## 1. Methodology & Initial Results

### Dataset Creation Approach and Baseline Model Selection
- The dataset (`synthetic_dataset_v1.json`) was created to cover a variety of business descriptions, simulating real-world use cases.
- A fine-tuned GPT-2 model (`sshleifer/tiny-gpt2`) is used as the baseline for domain name generation.
- The baseline model generates multiple domain name suggestions per business description.

### Initial Model Performance and Evaluation Metrics
- Each generated domain is evaluated using an LLM-based judge (OpenAI GPT-3.5-turbo) for three metrics: relevance, brandability, and safety.
- Results are saved in `eval_results.json` for further analysis.

## 2. Edge Case Analysis

### Discovery Process
- Edge cases are identified by analyzing the evaluation results in the notebook `02_edge_case_analysis.ipynb`.
- The process includes sorting and filtering domains by low scores and unsafe flags.

### Failure Taxonomy
- Failures are categorized as:
  - Unsafe (flagged by OpenAI Moderation API)
  - Low relevance (score < 5)
  - Low brandability (score < 5)
- Examples and counts for each category are provided in the notebook.

### Frequency Analysis
- The notebook computes the frequency of each failure type using value counts and summary statistics.

## 3. Iterative Improvement

### Improvement Strategies
- Iterative changes are made to the prompt, model parameters, and filtering logic to improve domain quality.
- Each change is documented in the notebook, with rationale for the adjustment.

### Quantified Results
- Before/after metrics are recorded for each iteration, showing improvements in relevance, brandability, and safety.

### LLM Judge Validation
- The LLM judge's consistency is checked by reviewing outputs and, if needed, manually inspecting edge cases.
- Fallback logic is implemented for API quota errors to ensure robust evaluation.

## 4. Model Comparison & Recommendations

### Performance Comparison
- Multiple model versions or parameter settings are compared using statistical analysis (e.g., mean, std, and significance tests in the notebook).

### Production Readiness
- The most robust and highest-performing model version is recommended for deployment, based on evaluation metrics and error handling.

### Future Improvements
- Suggestions include:
  - Expanding the dataset with more diverse business types
  - Further fine-tuning the model
  - Improving the LLM judge with ensemble or human-in-the-loop validation
  - Adding more granular failure categories

## API Development (Optional)
- The current project does not expose an API, but the codebase can be easily extended with a FastAPI or Flask endpoint for real-time domain evaluation.

## API Usage Examples

### Generate Domains (Safe Request)

**Request:**
```json
POST /generate-domains
{
  "business_description": "organic coffee shop in downtown area",
  "num_domains": 3
}
```
**Response:**
```json
{
  "suggestions": [
    {"domain": "organicbeanscafe.com", "confidence": 0.92},
    {"domain": "downtowncoffee.org", "confidence": 0.87},
    {"domain": "freshbreworganic.net", "confidence": 0.83}
  ],
  "status": "success"
}
```

### Generate Domains (Blocked/Unsafe Request)

**Request:**
```json
POST /generate-domains
{
  "business_description": "adult content website with explicit nude content",
  "num_domains": 3
}
```
**Response:**
```json
{
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

### Judge Domain (Example)

**Request:**
```json
POST /judge-domain
{
  "domain": "genzai.com",
  "description": "AI SaaS for Gen Z"
}
```
**Response:**
```json
{
  "relevance": 10,
  "brandability": 5,
  "safe": true
}
```

## Technologies Used

- **Python 3.10+** — Core programming language
- **PyTorch** — For running and fine-tuning the GPT-2 model
- **Transformers (Hugging Face)** — Model and tokenizer management
- **OpenAI API** — LLM-based domain scoring and moderation
- **tiktoken** — Token counting for prompt management
- **FastAPI** — REST API for domain generation and evaluation
- **Uvicorn** — ASGI server for running FastAPI
- **pandas** — Data analysis in notebooks
- **Jupyter Notebook** — For analysis, edge case exploration, and reporting
- **dotenv** — Environment variable management

---

**For details and code, see the scripts and notebooks in this repository.**
