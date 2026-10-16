from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import tiktoken
import json
import random
import re
import logging
from fastapi.middleware.cors import CORSMiddleware

# New import for OpenAI Python SDK v1+
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI client lazily-guarded: the API must still start when no
# key is configured (e.g. local dev). Endpoints that need OpenAI return a
# clear 503 instead of crashing the whole app at import time.
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        # e.g. broken proxy env vars (httpx) on some machines
        print(f"⚠️ OpenAI client unavailable, continuing without it: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_model():
    """Load the domain-suggester model, with graceful fallbacks.

    Tries, in order:
      1. Fine-tuned model from Hugging Face Hub
      2. Local fine-tuned model in ``local_model_finetuned/`` (if present)
      3. Base ``gpt2`` from Hugging Face Hub

    Returns ``(tokenizer, model, device)``; ``(None, None, device)`` if every
    source fails (e.g. no network). Never raises, so importing this module
    and starting the API server never crashes on model loading.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    candidates = [
        ("Hugging Face Hub (fine-tuned)", "Octopus87/domain-suggester-gpt2", True),
        ("local model", os.path.abspath(os.path.join(os.path.dirname(__file__), "../local_model_finetuned")), False),
        ("Hugging Face Hub (base gpt2)", "gpt2", True),
    ]
    for label, source, is_hub in candidates:
        if not is_hub and not os.path.isdir(source):
            logger.info(f"⏭️  Skipping {label}: directory not found: {source}")
            continue
        try:
            logger.info(f"🚀 Loading model ({label})...")
            tokenizer = AutoTokenizer.from_pretrained(
                source, token=HF_TOKEN if is_hub else None, trust_remote_code=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                source,
                token=HF_TOKEN if is_hub else None,
                # Render free/Starter tiers have only 512 MiB RAM: fp32
                # weights (~500 MB) OOM the instance during load. fp16
                # halves weights to ~250 MB; GPT-2 generation quality is
                # roughly unchanged. low_cpu_mem_usage avoids a second
                # full fp32 copy in RAM during loading.
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            logger.info(f"✅ Model loaded successfully ({label}) on {device}")
            return tokenizer, model, device
        except Exception as e:
            logger.warning(f"❌ Error loading model ({label}): {e}")

    logger.error("❌ All model sources failed — /generate-domains will return 503.")
    return None, None, device


tokenizer, model, device = _load_model()

# Initialize FastAPI app
app = FastAPI(
    title="Domain Suggester API",
    description="AI-powered domain name suggestions for businesses",
    version="1.0.0"
)

# Allow frontend CORS - more permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# Request Models
class DomainRequest(BaseModel):
    business_description: str
    num_domains: int = 3

class JudgeRequest(BaseModel):
    domain: str
    description: str

@app.get("/")
@app.head("/")
def root():
    return {"message": "✅ Domain Suggestion API is live", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
@app.head("/health")
def health_check():
    return {"status": "healthy", "timestamp": "2025-01-07", "service": "domain-suggester"}

@app.post("/generate-domains")
def generate_domains_api(req: DomainRequest):
    # Validate number of domains requested
    if req.num_domains < 1:
        raise HTTPException(status_code=400, detail="Number of domains must be at least 1")
    if req.num_domains > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 domains can be requested at once")

    # Validate business description
    if len(req.business_description.strip()) == 0:
        raise HTTPException(status_code=400, detail="Business description cannot be empty")
    if len(req.business_description) > 500:
        raise HTTPException(status_code=400, detail="Business description too long (max 500 characters)")

    # Model must have loaded for generation to work
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Domain model is unavailable (could not load from Hugging Face Hub or local_model_finetuned/). Check network access and restart the server.",
        )

    # Safety check using OpenAI Moderation (new API). Skipped when no
    # OpenAI client is configured; generation still works.
    flagged = False
    if client is not None:
        try:
            mod = client.moderations.create(
                model="omni-moderation-latest",
                input=req.business_description
            )
            flagged = mod.results[0].flagged
        except Exception as e:
            flagged = False
            logger.warning(f"Moderation fallback: {e}")
    else:
        logger.info("Moderation skipped: OPENAI_API_KEY not configured")

    if flagged:
        return {
            "suggestions": [],
            "status": "blocked",
            "message": "Request contains inappropriate content"
        }

    # Training-format prompt (see docs/technical_implementation.md §4):
    # training examples were
    # "Generate domain names for: {desc}\nSuggestions: {list}<|endoftext|>".
    # The old instruction-style prompt shared no surface form with the
    # training data, so the fine-tuned model fell back to generic GPT-2
    # continuation (off-topic gibberish). Keep the prompt EXACTLY in
    # training format; the model continues after "Suggestions: " with a
    # comma-separated list, then EOS.
    # Same prompt / sampling as gradio-space/app.py.
    prompt = f"Generate domain names for: {req.business_description}\nSuggestions: "
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate domain names.")

    # Remove prompt if echoed
    if decoded.startswith(prompt):
        domains_text = decoded[len(prompt):].strip()
    else:
        domains_text = decoded.strip()

    # The list ends at EOS or at the first newline; anything after a
    # newline is the model wandering into a new (untrained) continuation.
    domains_text = domains_text.split("\n")[0].strip()

    logger.info(f"🧠 Model output: {domains_text}")

    # Extract domain names
    domain_pattern = r"[a-zA-Z0-9-]{3,}\.(?:com|org|net)"
    found_domains = re.findall(domain_pattern, domains_text)

    if len(found_domains) < req.num_domains:
        candidates = [d.strip() for d in domains_text.split(",") if d.strip()]
        more_domains = [d for d in candidates if re.fullmatch(domain_pattern, d)]
        for d in more_domains:
            if d not in found_domains:
                found_domains.append(d)

    unique_domains = []
    seen = set()
    for d in found_domains:
        if d not in seen:
            seen.add(d)
            unique_domains.append(d)

    if len(unique_domains) < req.num_domains:
        candidates = [d.strip() for d in domains_text.split(",") if d.strip()]
        for d in candidates:
            if d not in unique_domains:
                unique_domains.append(d)
            if len(unique_domains) >= req.num_domains:
                break

    suggestions = [
        {"domain": d, "confidence": round(random.uniform(0.8, 0.99), 2)}
        for d in unique_domains[:req.num_domains]
    ]
    return {"suggestions": suggestions, "status": "success"}

@app.post("/judge-domain")
def judge_domain_api(req: JudgeRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Domain judging requires OPENAI_API_KEY. Set it in a .env file and restart the server.",
        )

    system_msg = "You are a helpful assistant that scores domain names."
    user_prompt = f"""
Business description: \"{req.description}\"
Domain name: \"{req.domain}\"

Please rate the domain on:
- Relevance (0-10): How well does the domain reflect the business description?
- Brandability (0-10): How catchy, memorable, and professional is the domain?
- Safety (true/false): Is the domain name safe and appropriate (no offensive content)?

Return your answer in strict JSON format, for example:
{{"relevance": 8, "brandability": 7, "safe": true}}
"""

    try:
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens_in_prompt = len(encoder.encode(system_msg + user_prompt))
        logger.info(f"🧮 Tokens in prompt: {tokens_in_prompt}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=150,
        )
        # OpenAI Python SDK v1+: message is an object, not a dict
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"OpenAI Judge error: {e}")
        raise HTTPException(
            status_code=502,
            detail="Domain judging is temporarily unavailable (OpenAI request failed).",
        )
