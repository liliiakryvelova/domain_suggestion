# scripts/eval_llm_judge.py

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
from pathlib import Path
import tiktoken

# --- Load keys and models ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

MODEL_PATH = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# --- Content Safety Check ---
def is_safe(text: str) -> bool:
    try:
        response = openai.Moderation.create(input=text)
        flagged = response["results"][0]["flagged"]
        categories = response["results"][0]["categories"]
        if flagged:
            print(f"❌ Flagged: {text} \nCategories: {categories}")
        return not flagged
    except Exception as e:
        print(f"Moderation error: {e}")
        return False


# --- Domain Generation ---
def generate_domains(prompt: str, num_domains=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=num_domains,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(o, skip_special_tokens=True).replace(prompt, "").strip()
        for o in outputs
    ]


# --- Domain Scoring via OpenAI ---
def judge_domain(domain: str, description: str):
    system_msg = "You are a helpful assistant that scores domain names."
    user_prompt = f"""
Business description: \"{description}\"
Domain name: \"{domain}\"

Please rate the domain on:
- Relevance (0-10): How well does the domain reflect the business description?
- Brandability (0-10): How catchy, memorable, and professional is the domain?
- Safety (true/false): Is the domain name safe and appropriate (no offensive content)?

Return your answer in strict JSON format, for example:
{{"relevance": 8, "brandability": 7, "safe": true}}
"""
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


# --- Main Evaluation Pipeline ---
def main():
    with open("data/synthetic_dataset_v1.json", "r") as f:
        data = json.load(f)

    eval_results = []

    for item in data[:10]:  # Adjust slice for full run
        desc = item["business_description"]

        if not is_safe(desc):
            print("🚫 Skipping unsafe business description.")
            continue

        prompt = f"Suggest domain names for: {desc}"
        domains = generate_domains(prompt)

        for d in domains:
            if not is_safe(d):
                print("⚠️ Skipping unsafe domain.")
                continue

            try:
                scores = judge_domain(d, desc)
            except Exception as e:
                print(f"🛑 Error judging domain '{d}': {e}")
                scores = {"relevance": None, "brandability": None, "safe": None}

            eval_results.append({
                "business_description": desc,
                "domain": d,
                **scores,
            })

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print("✅ Evaluation complete. Results saved to data/eval_results.json")


if __name__ == "__main__":
    main()
