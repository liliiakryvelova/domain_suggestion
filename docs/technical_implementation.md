# Complete Technical Implementation Guide

## Overview
This document provides comprehensive technical details for implementing, training, and deploying the Domain Suggester GPT-2 model.

## Architecture Deep Dive

### Model Pipeline Architecture
```
Input Business Description
         ↓
    Text Preprocessing
         ↓
    GPT-2 Generation
         ↓
    Post-Processing Filter
         ↓
    Quality Scoring
         ↓
    Safety Validation
         ↓
    Ranked Domain Suggestions
```

### Core Components

#### 1. Input Processing Module
```python
class InputProcessor:
    def __init__(self):
        self.max_length = 256
        self.business_categories = self._load_categories()
        
    def preprocess(self, business_description: str) -> str:
        """
        Clean and format business description for optimal generation
        """
        # Normalize text
        description = business_description.strip().lower()
        
        # Add context markers
        prompt = f"Generate domain names for: {description}"
        
        # Add category hint if detected
        category = self._detect_category(description)
        if category:
            prompt += f" (Category: {category})"
            
        return prompt
    
    def _detect_category(self, description: str) -> str:
        """Detect business category for better context"""
        category_keywords = {
            'tech': ['ai', 'software', 'app', 'platform', 'saas'],
            'retail': ['store', 'shop', 'marketplace', 'ecommerce'],
            'services': ['consulting', 'agency', 'firm', 'service'],
            'healthcare': ['medical', 'health', 'clinic', 'therapy'],
            'education': ['learning', 'education', 'training', 'course']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description for keyword in keywords):
                return category
        return None
```

#### 2. Generation Engine
```python
class DomainGenerator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def generate_domains(self, prompt: str, num_domains: int = 5) -> List[str]:
        """
        Generate domain suggestions using multiple strategies
        """
        all_domains = []
        
        # Strategy 1: High creativity (temperature = 1.0)
        creative_domains = self._generate_with_params(
            prompt, num_domains//2, temperature=1.0, top_p=0.9
        )
        all_domains.extend(creative_domains)
        
        # Strategy 2: Conservative (temperature = 0.7)
        conservative_domains = self._generate_with_params(
            prompt, num_domains//2, temperature=0.7, top_p=0.95
        )
        all_domains.extend(conservative_domains)
        
        # Strategy 3: Beam search for quality
        beam_domains = self._generate_with_beam_search(prompt, num_domains//2)
        all_domains.extend(beam_domains)
        
        # Remove duplicates and clean
        unique_domains = list(set(self._extract_domains(all_domains)))
        return unique_domains[:num_domains * 2]  # Over-generate for filtering
    
    def _generate_with_params(self, prompt: str, num_domains: int, 
                             temperature: float, top_p: float) -> List[str]:
        """Generate with specific parameters"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                num_return_sequences=num_domains,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        return generated_texts
    
    def _generate_with_beam_search(self, prompt: str, num_domains: int) -> List[str]:
        """Generate using beam search for higher quality"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=25,
                num_beams=8,
                num_return_sequences=num_domains,
                early_stopping=True,
                repetition_penalty=1.1,
                length_penalty=0.8,  # Prefer shorter domains
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        return generated_texts
    
    def _extract_domains(self, generated_texts: List[str]) -> List[str]:
        """Extract valid domain names from generated text"""
        domains = []
        domain_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.(com|net|org|io|co)\b'
        
        for text in generated_texts:
            matches = re.findall(domain_pattern, text, re.IGNORECASE)
            for match in matches:
                domain = match[0] + '.' + match[1] if isinstance(match, tuple) else match
                domains.append(domain.lower())
                
        return domains
```

#### 3. Quality Scoring System
```python
class QualityScorer:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def score_domain(self, domain: str, business_description: str) -> Dict[str, float]:
        """
        Comprehensive domain quality scoring
        """
        scores = {}
        
        # 1. Relevance scoring (LLM-based)
        scores['relevance'] = self._score_relevance(domain, business_description)
        
        # 2. Brandability scoring (rule-based + LLM)
        scores['brandability'] = self._score_brandability(domain)
        
        # 3. Technical quality
        scores['technical'] = self._score_technical_quality(domain)
        
        # 4. Safety assessment
        scores['safety'] = self._assess_safety(domain + " " + business_description)
        
        # 5. Overall score (weighted combination)
        scores['overall'] = (
            scores['relevance'] * 0.4 +
            scores['brandability'] * 0.3 +
            scores['technical'] * 0.2 +
            (10.0 if scores['safety'] else 0.0) * 0.1
        )
        
        return scores
    
    def _score_relevance(self, domain: str, business_description: str) -> float:
        """Score domain relevance using GPT-4"""
        prompt = f"""
        Rate the relevance of this domain name for the given business on a scale of 1-10.
        
        Business: {business_description}
        Domain: {domain}
        
        Consider:
        - How well the domain reflects the business purpose
        - Industry appropriateness
        - Target audience alignment
        
        Respond with only a number from 1-10.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score = float(response.choices[0].message.content.strip())
            return max(1.0, min(10.0, score))  # Clamp to 1-10
        except:
            return 5.0  # Default score if API fails
    
    def _score_brandability(self, domain: str) -> float:
        """Score brandability using multiple factors"""
        score = 10.0
        
        # Length penalty (optimal 6-12 characters)
        base_domain = domain.split('.')[0]
        length = len(base_domain)
        if length < 4:
            score -= 2.0  # Too short
        elif length > 15:
            score -= (length - 15) * 0.5  # Too long
        elif length > 12:
            score -= (length - 12) * 0.2  # Slightly long
            
        # Pronounceability (vowel/consonant ratio)
        vowels = sum(1 for c in base_domain.lower() if c in 'aeiou')
        consonants = sum(1 for c in base_domain.lower() if c.isalpha() and c not in 'aeiou')
        if consonants > 0:
            vowel_ratio = vowels / consonants
            if vowel_ratio < 0.2 or vowel_ratio > 2.0:
                score -= 1.5  # Hard to pronounce
                
        # Hyphen penalty
        if '-' in base_domain:
            score -= 1.0
            
        # Number penalty
        if any(c.isdigit() for c in base_domain):
            score -= 0.5
            
        # Dictionary word bonus
        if self._contains_dictionary_words(base_domain):
            score += 1.0
            
        return max(1.0, min(10.0, score))
    
    def _score_technical_quality(self, domain: str) -> float:
        """Score technical aspects of the domain"""
        score = 10.0
        
        # Valid format check
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}$', domain):
            score -= 5.0
            
        # TLD preference (.com gets bonus)
        tld = domain.split('.')[-1].lower()
        if tld == 'com':
            score += 0.5
        elif tld in ['net', 'org']:
            pass  # Neutral
        else:
            score -= 1.0  # Less preferred TLD
            
        return max(1.0, min(10.0, score))
    
    def _assess_safety(self, text: str) -> bool:
        """Check content safety using OpenAI Moderation"""
        try:
            response = self.openai_client.moderations.create(input=text)
            return not response.results[0].flagged
        except:
            return True  # Default to safe if API fails
```

#### 4. Training Pipeline
```python
class ModelTrainer:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare training dataset"""
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Format training examples
        formatted_examples = []
        for item in raw_data:
            business_desc = item['business_description']
            domains = item['expected_domains']
            
            # Create training text
            domain_list = ", ".join(domains)
            text = f"Generate domain names for: {business_desc}\nSuggestions: {domain_list}<|endoftext|>"
            formatted_examples.append({"text": text})
        
        dataset = Dataset.from_list(formatted_examples)
        return dataset.train_test_split(test_size=0.1)
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
    
    def train(self, dataset_path: str, output_dir: str):
        """Train the model"""
        # Prepare data
        dataset = self.prepare_dataset(dataset_path)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Set labels for causal LM
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
```

## Deployment Architecture

### FastAPI Application Structure
```python
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os

app = FastAPI(
    title="Domain Suggester API",
    description="AI-powered domain name suggestions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure for production
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global components
domain_generator = None
quality_scorer = None

@app.on_event("startup")
async def startup_event():
    global domain_generator, quality_scorer
    domain_generator = DomainGenerator("./local_model_finetuned")
    quality_scorer = QualityScorer()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/generate-domains")
@limiter.limit("10/minute")
async def generate_domains(request: Request, payload: DomainRequest):
    """Generate domain suggestions with quality scoring"""
    try:
        # Input validation and safety check
        if not quality_scorer._assess_safety(payload.business_description):
            raise HTTPException(
                status_code=400, 
                detail="Content flagged by safety filters"
            )
        
        # Generate domains
        domains = domain_generator.generate_domains(
            payload.business_description, 
            payload.num_domains * 2  # Over-generate for filtering
        )
        
        # Score and filter domains
        scored_domains = []
        for domain in domains:
            scores = quality_scorer.score_domain(domain, payload.business_description)
            
            # Filter low-quality domains
            if scores['overall'] >= 6.0 and scores['safety']:
                scored_domains.append({
                    "domain": domain,
                    "confidence": scores['overall'] / 10.0,
                    "relevance": scores['relevance'],
                    "brandability": scores['brandability'],
                    "safe": scores['safety']
                })
        
        # Sort by overall score and return top N
        scored_domains.sort(key=lambda x: x['confidence'], reverse=True)
        final_suggestions = scored_domains[:payload.num_domains]
        
        return {
            "suggestions": final_suggestions,
            "status": "success",
            "total_generated": len(domains),
            "after_filtering": len(scored_domains)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Production Deployment

#### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download/prepare model (in production, use model registry)
RUN python scripts/download_model.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

#### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: domain-suggester
spec:
  replicas: 3
  selector:
    matchLabels:
      app: domain-suggester
  template:
    metadata:
      labels:
        app: domain-suggester
    spec:
      containers:
      - name: domain-suggester
        image: domain-suggester:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: domain-suggester-service
spec:
  selector:
    app: domain-suggester
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Performance Optimization

### Model Optimization
```python
# Model optimization techniques
class OptimizedDomainGenerator(DomainGenerator):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
        # Enable optimizations
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.half()  # FP16 inference
            
        # Torch JIT compilation
        self.model = torch.jit.trace(
            self.model, 
            example_inputs
        )
        
        # ONNX conversion for even faster inference
        self._convert_to_onnx()
    
    def _convert_to_onnx(self):
        """Convert model to ONNX for optimized inference"""
        torch.onnx.export(
            self.model,
            example_inputs,
            "model.onnx",
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
```

### Caching Strategy
```python
from functools import lru_cache
import redis

class CachedDomainGenerator:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.generator = DomainGenerator("./model")
        
    @lru_cache(maxsize=1000)
    def generate_domains_cached(self, business_description: str, num_domains: int):
        """Cache domain generations in memory"""
        cache_key = f"domains:{hash(business_description)}:{num_domains}"
        
        # Check Redis cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Generate if not cached
        result = self.generator.generate_domains(business_description, num_domains)
        
        # Cache for 1 hour
        self.redis_client.setex(
            cache_key, 
            3600, 
            json.dumps(result)
        )
        
        return result
```

This comprehensive technical implementation provides all the necessary details for building, training, and deploying a production-ready domain suggestion system.
