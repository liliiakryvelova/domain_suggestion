import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load eval results
with open("data/eval_results.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Only proceed if data is present and valid
if df.empty or not all(col in df.columns for col in ["relevance", "brandability", "safe"]):
    raise ValueError("❌ eval_results.json is empty or missing required fields")

# Clean and preprocess
df["safe"] = df["safe"].fillna(False)
df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
df["brandability"] = pd.to_numeric(df["brandability"], errors="coerce")
df.dropna(subset=["relevance", "brandability"], inplace=True)

# Categorize failures
def categorize(row):
    if not row["safe"]:
        return "Unsafe"
    elif row["relevance"] < 5 and row["brandability"] < 5:
        return "Irrelevant & Unbrandable"
    elif row["relevance"] < 5:
        return "Low Relevance"
    elif row["brandability"] < 5:
        return "Low Brandability"
    else:
        return "Good"

df["category"] = df.apply(categorize, axis=1)

# Save all failure cases
failures = df[df["category"] != "Good"]
failures["failure_reason"] = failures["category"]  # for clarity
failures.to_csv("data/failure_cases.csv", index=False)

# Summary counts
print("✅ Failure Reason Breakdown:")
print(failures["failure_reason"].value_counts())

# Metric: pass rate
def calc_pass_rate(df):
    total = len(df)
    passed = (df["category"] == "Good").sum()
    return round((passed / total) * 100, 2)

pass_rate = calc_pass_rate(df)
print(f"\n✅ Pass rate: {pass_rate}%")

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["relevance"], bins=10, kde=True)
plt.title("Relevance Score Distribution")

plt.subplot(1, 2, 2)
sns.histplot(df["brandability"], bins=10, kde=True)
plt.title("Brandability Score Distribution")

plt.suptitle("Domain Evaluation Score Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("data/score_distributions.png")
plt.show()

# Save overall summary
summary = {
    "total_evaluated": len(df),
    "total_failures": len(failures),
    "pass_rate_percent": pass_rate,
    "failure_reason_counts": failures["failure_reason"].value_counts().to_dict()
}

with open("data/failure_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Analysis complete. Failures saved to data/failure_cases.csv")
