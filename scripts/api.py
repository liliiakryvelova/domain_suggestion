from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import os
from dotenv import load_dotenv
import tiktoken
import json
import random
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

MODEL_PATH = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://liliiakryvelova.github.io"],  # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DomainRequest(BaseModel):
    business_description: str
    num_domains: int = 3

class JudgeRequest(BaseModel):
    domain: str
    description: str

@app.post("/generate-domains")
def generate_domains_api(req: DomainRequest):
    # Safety check using OpenAI Moderation
    try:
        mod = openai.Moderation.create(input=req.business_description)
        flagged = mod["results"][0]["flagged"]
    except Exception as e:
        flagged = False
    if flagged:
        return {
            "suggestions": [],
            "status": "blocked",
            "message": "Request contains inappropriate content"
        }
    prompt = f"Suggest domain names for: {req.business_description}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=req.num_domains,
        pad_token_id=tokenizer.eos_token_id,
    )
    domains = [tokenizer.decode(o, skip_special_tokens=True).replace(prompt, "").strip() for o in outputs]
    # Assign random confidence scores for demo
    suggestions = [
        {"domain": d, "confidence": round(random.uniform(0.8, 0.99), 2)} for d in domains
    ]
    return {"suggestions": suggestions, "status": "success"}

@app.post("/judge-domain")
def judge_domain_api(req: JudgeRequest):
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
        print(f"🧮 Tokens in prompt: {tokens_in_prompt}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=150,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error: {e}")
        # Fallback mock
        return {
            "relevance": random.randint(0, 10),
            "brandability": random.randint(0, 10),
            "safe": random.choice([True, False]),
        }
