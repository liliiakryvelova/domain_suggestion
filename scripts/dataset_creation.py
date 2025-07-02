import json
import random
from faker import Faker
from pathlib import Path

fake = Faker()

business_types = [
    "organic coffee shop",
    "AI SaaS startup",
    "family law firm",
    "animal shelter nonprofit",
    "online education platform",
    "eco-friendly clothing brand",
    "gaming mobile app",
    "luxury skincare line",
    "real estate agency",
    "pet grooming service",
    "vegan bakery",
    "mental health counseling center"
]

location_modifiers = [
    "in downtown area",
    "in San Francisco",
    "with a global audience",
    "for Gen Z customers",
    "in a small town",
    "targeting remote workers",
    "with a luxury touch",
    "in a beachside city"
]

def generate_description():
    business = random.choice(business_types)
    location = random.choice(location_modifiers)
    return f"{business} {location}"

def generate_domain_name(description: str) -> str:
    words = description.lower().replace("-", "").split()
    core = ''.join(random.sample(words, min(3, len(words))))
    domain = f"{core}{random.choice(['.com', '.org', '.net'])}"
    return domain.replace(" ", "")

def generate_dataset(n_samples=200):
    dataset = []
    for _ in range(n_samples):
        desc = generate_description()
        domains = list({generate_domain_name(desc) for _ in range(3)})
        dataset.append({
            "business_description": desc,
            "expected_domains": domains
        })
    return dataset

def save_dataset(data, file_path="data/synthetic_dataset_v1.json"):
    Path("data").mkdir(exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
    print(f"✅ Dataset saved to data/synthetic_dataset_v1.json with {len(dataset)} entries.")
