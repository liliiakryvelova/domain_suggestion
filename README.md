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
- Edge cases are identified by analyzing evaluation results in the notebook `02_edge_case_analysis.ipynb`.
- This involves sorting and filtering domains by low scores and unsafe flags.

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
- Iterative improvements are made to the prompt, model parameters, and filtering logic to enhance domain quality.
- Each change is documented with rationale in the notebook.

### Quantified Results
- Before/after metrics record improvements in relevance, brandability, and safety.

### LLM Judge Validation
- Consistency is checked by reviewing outputs and manually inspecting edge cases.
- Fallback logic is implemented for API quota errors to ensure robust evaluation.

## 4. Model Comparison & Recommendations

### Performance Comparison
- Multiple model versions or parameter settings are compared using statistical analysis.

### Production Readiness
- The most robust and highest-performing model version is recommended for deployment, based on evaluation and error handling.

### Future Improvements
- Suggestions include:
  - Expanding the dataset with more diverse business types
  - Further fine-tuning the model
  - Improving the LLM judge with ensemble or human-in-the-loop validation
  - Adding more granular failure categories

## 5. API Development & Usage

### Overview
- The project exposes a FastAPI server wrapping the fine-tuned GPT-2 model for domain suggestion.
- The API also integrates OpenAI moderation and an LLM judge (GPT-3.5-turbo) for domain evaluation.

### Generate Domains (Safe Request)

**Request:**
```json
POST /generate-domains
{
  "business_description": "organic coffee shop in downtown area",
  "num_domains": 3
}
