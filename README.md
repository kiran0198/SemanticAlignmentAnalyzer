# SemanticAlignmentAnalyzer

# Semantic Alignment Analyzer

## Overview

Semantic Alignment Analyzer is a multimodal AI system that measures whether an image and a piece of text convey the same semantic meaning.

Instead of comparing raw pixels with raw words, the system first extracts semantic themes from both modalities and then computes a Semantic Alignment Score (SAAS).

Example:

Image:

* Football match

Text:

* "Football is my favorite game"

Output:

```json
{
  "image_theme": "sports",
  "text_theme": "sports",
  "theme_match": true,
  "saas": 98.37
}
```

---

## Problem Statement

Traditional sentiment analysis focuses only on text.

Many real-world posts contain both:

* Images
* Text

A meaningful multimodal system should determine whether both modalities describe the same semantic concept.

This project explores semantic consistency between images and text.

---

## Project Pipeline

Image
→ ResNet18 Theme Extractor
→ Image Theme

Text
→ DistilBERT Theme Extractor
→ Text Theme

Image Theme + Text Theme
→ Semantic Alignment Score (SAAS)

---

## Models Used

### Image Theme Extractor

Architecture:

* ResNet18
* Transfer Learning

Output Classes:

* Animals
* Food
* Sports
* Plants
* Famous Places

Validation Accuracy:

~94.7%

---

### Text Theme Extractor

Architecture:

* DistilBERT
* Custom Projection Head

Output Classes:

* Animals
* Food
* Sports
* Plants
* Landmarks

Validation Accuracy:

100% on the synthetic theme dataset.

---

## Mathematical Formulation

Let:

I = Image Theme

T = Text Theme

If:

I = T

Then:

SAAS = High

Otherwise:

SAAS = Low

Current implementation:

SAAS = ((ImageConfidence + TextConfidence) / 2) × 100

for matching themes.

For mismatching themes, a penalty function produces a low alignment score.

---

## Datasets Used

### Animal Images Dataset

Source:
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

Credit:
Sourav Banerjee

---

### Indian Food Images Dataset

Source:
https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset

Credit:
Sourav Banerjee

---

### Sports Image Dataset

Source:
https://www.kaggle.com/datasets/rishikeshkonapure/sports-image-dataset

Credit:
Rishikesh Konapure

---

### Plants Type Dataset

Source:
https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets

Credit:
Yudha Islam Isulistya

---

### Famous Places Dataset

Source:
https://www.kaggle.com/datasets/ilyaryabov/pictures-of-famous-places

Credit:
Ilya Ryabov

---

## Repository Structure

```text
SemanticAlignmentAnalyzer/

├── final_demo.py

├── models/
│   ├── theme_extractor.pth
│   └── text_encoder.pth

├── sample_images/

├── results/

├── README.md

└── requirements.txt
```

---

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python final_demo.py
```

Example Output:

```json
{
  "image_theme": "sports",
  "text_theme": "sports",
  "theme_match": true,
  "saas": 98.37
}
```
SAAS = Semantic Alignment Analysis Score
---

## Current Limitations

1. Theme-based semantic comparison.

The current system compares extracted themes rather than learning a joint image-text embedding space.

2. Limited semantic categories.

Only five semantic themes are supported.

3. Synthetic text dataset.

The text training dataset was generated synthetically.

4. No end-to-end multimodal contrastive learning.

The image encoder and text encoder are trained independently.

---

## Future Work

* Joint image-text contrastive learning (CLIP-style architecture)
* Larger semantic vocabulary
* Fine-grained semantic alignment
* Cross-modal retrieval
* Explainable semantic reasoning

---

## Author

Kiran Kumar Sahu

M.Tech Computer Science & Engineering

OP Jindal University

Research Interest:
Multimodal AI, Semantic Analysis, Machine Learning
