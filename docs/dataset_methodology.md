# Dataset Creation Methodology

## Overview
This document details the methodology used to create the training dataset for the Domain Suggester GPT-2 model.

## Dataset Composition

### 1. Synthetic Data Generation Strategy

**Current Approach:**
- **Base Templates**: 12 predefined business types covering diverse industries
- **Location Modifiers**: 8 geographic and demographic modifiers
- **Combinatorial Generation**: Random combination of business types and modifiers

**Business Categories Covered:**
```python
business_types = [
    "organic coffee shop",           # Food & Beverage
    "AI SaaS startup",              # Technology
    "family law firm",              # Professional Services
    "animal shelter nonprofit",      # Non-profit
    "online education platform",    # EdTech
    "eco-friendly clothing brand",   # Fashion/Retail
    "gaming mobile app",            # Entertainment
    "luxury skincare line",         # Beauty/Health
    "real estate agency",           # Real Estate
    "pet grooming service",         # Services
    "vegan bakery",                 # Food & Beverage
    "mental health counseling center" # Healthcare
]
```

### 2. Data Quality Assurance

**Domain Name Generation Logic:**
1. **Word Extraction**: Extract core words from business description
2. **Combination Strategy**: Sample 1-3 words randomly
3. **TLD Selection**: Random choice from .com, .org, .net
4. **Deduplication**: Ensure 3 unique domains per business description

**Limitations Identified:**
- Limited business type diversity (only 12 categories)
- Simplistic domain generation (may not reflect real branding strategies)
- No validation against existing domains
- No consideration for trademark conflicts

### 3. Recommended Improvements

**Enhanced Dataset Strategy:**
1. **Expand Business Categories**: Include 50+ business types
2. **Real Domain Analysis**: Incorporate analysis of successful real domains
3. **Branding Principles**: Apply actual domain naming conventions
4. **Market Research**: Include trending business models
5. **Quality Validation**: Manual review of generated pairs

**Implementation Plan:**
```python
def enhanced_dataset_creation():
    # 1. Load real domain examples from successful businesses
    # 2. Apply branding heuristics (short, memorable, brandable)
    # 3. Include domain availability checking
    # 4. Add quality scoring during generation
    # 5. Implement stratified sampling across industries
```

## Dataset Statistics

**Current Dataset (v1):**
- **Size**: 200 business descriptions
- **Domains per description**: 3 unique domains
- **Total domain examples**: ~600 domain-business pairs
- **Industry coverage**: 12 categories
- **Geographic diversity**: 8 location types

**Target Dataset (v2 - Recommended):**
- **Size**: 2,000+ business descriptions
- **Domains per description**: 5-10 high-quality domains
- **Industry coverage**: 50+ categories
- **Quality validation**: Manual review + automated scoring
- **Real-world validation**: Domain availability checks
