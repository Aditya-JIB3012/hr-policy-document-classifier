# HR Policy Document Classifier

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7cec7b2dmd/badge.svg)](https://ask.thili.ai/b/7cec7b2dmd)

The HR Policy Document Classifier is a tool designed for HR administrators to automate the classification of internal policy documents. By fine-tuning a GPT-2 model, it assigns categories like `<leave_policy>` or `<expense_claim>` to document sections, streamlining retrieval and departmental routing.

## What you are building

**Goal.** You will build a tool that automatically classifies HR policy document paragraphs into categories like 'leave policy' or 'expense claim', enabling quick retrieval and departmental routing for HR administrators. This project demonstrates the fine-tuning logic and inference pipeline for text classification.

**Done when.** The Streamlit app allows users to upload a text file, and it successfully displays classified sections of the document, with each paragraph correctly assigned to one of the predefined HR categories.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect example HR document paragraphs and assign 3-5 categories.**  
      30 curated HR paragraphs across 5 categories: leave_policy, expense_claim, code_of_conduct, remote_work, health_safety.
- [x] **2. Extend the GPT-2 tokenizer and model for these HR categories.**  
      Added 5 special tokens (`<leave_policy>`, `<expense_claim>`, etc.) and resized model embeddings.
- [x] **3. Train the model using the prepared dataset and Hugging Face Trainer.**  
      Full training pipeline using DataCollatorForLanguageModeling and HF Trainer with configurable epochs/steps.
- [x] **4. Implement a function to classify new document text.**  
      Inference pipeline that splits documents into paragraphs, computes logits over category tokens, and returns predictions with confidence scores.
- [x] **5. Create a Streamlit app to upload a text file and display classified sections.**  
      Full-featured Streamlit UI with file upload, direct text input, sample templates, departmental routing, confidence filtering, and JSON/text export.

## Run it

You need Python 3.12+ and git.

```bash
# Windows
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py

# macOS / Linux
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

Run the tests:
```bash
pytest
```

Run the full evaluation benchmark:
```bash
python evaluate.py
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`.

## Evaluation & Baseline Results

Model evaluation was performed by comparing pre-fine-tuning baseline performance against post-fine-tuning performance on a **held-out test set of 15 previously unseen HR policy paragraphs** (3 per category). Full metrics are saved in `results/evaluation_report.json` and `results/evaluation_results.md`.

### Results Summary

| Metric | Pre-Training Baseline | Post-Training (3 Epochs) | Change |
|---|---|---|---|
| **Training Loss** | N/A | **3.9963** (down from 4.829) | -0.83 loss |
| **Held-out Test Set Accuracy** | 26.7% (4/15) | **33.3%** (5/15) | +6.6% improvement |
| **Training Set Accuracy** | 20.0% (random) | **36.7%** (11/30) | +16.7% improvement |

### Per-Category Accuracy on Held-Out Test Data

| Category | Test Accuracy | Status | Key Pattern Identified |
|---|---|---|---|
| `leave_policy` | **100.0% (3/3)** | Strong | Accurately tags statutory leave, sick leave, jury duty |
| `expense_claim` | **66.7% (2/3)** | Moderate | Detects conferences, per diems, taxi receipts |
| `code_of_conduct` | **0.0% (0/3)** | Underfitted | Needs more training epochs/examples to differentiate from ethics |
| `remote_work` | **0.0% (0/3)** | Underfitted | Confused with expense claims (stipends) or leave |
| `health_safety` | **0.0% (0/3)** | Underfitted | Overlaps with workplace conduct/leave definitions |

## Performance, Limitations & Honest Assessment

### What Works Well:
- **Demonstration of the End-to-End Pipeline:** Tokenizer extension, vocabulary expansion, embedding matrix resizing, Causal LM fine-tuning with HF Trainer, and next-token logit extraction are fully working and verified by unit tests.
- **Top Categories:** Common categories like `leave_policy` and `expense_claim` show strong classification accuracy on novel phrasing.
- **Interactive Streamlit UI:** Handles arbitrary `.txt` and `.md` file uploads, presents clear visual tags with confidence bars, provides departmental routing info, and exports results to JSON.

### Limitations & Where It Falls Short:
- **Small Training Sample Size:** The training dataset contains 30 samples (6 per category). Neural language models typically require hundreds of examples per class to generalize reliably across subtle semantic distinctions.
- **Generative vs. Discriminative Architecture:** GPT-2 is a decoder-only causal language model. Repurposing next-token generation for classification works well as a conceptual drill, but purpose-built encoders (e.g. RoBERTa or DeBERTa) with a classification head natively achieve higher discriminative power.
- **Confidence Calibration:** The confidence values are softmax distributions across the 5 category tokens. With limited fine-tuning steps, the raw probabilities cluster closely around ~0.20-0.25 and require calibration.
- **Hardware & Epoch Constraints:** Running on CPU requires keeping training epochs low (1-3 epochs) for interactive responsiveness. Higher epochs on GPU would substantially improve classification convergence across all 5 classes.

### Potential Improvements:
1. **Expand Data:** Scale dataset to 100+ paragraphs per category sourced from standard employee handbooks.
2. **Encoder Head:** Implement a classification head (`AutoModelForSequenceClassification` with BERT/RoBERTa) for comparison.
3. **Temperature Scaling:** Add temperature scaling to calibrate confidence probabilities.
4. **Data Augmentation:** Apply back-translation and synonym replacement to synthesize diverse phrasing.

## HR Categories & Departmental Routing

| Category | Tag | Routed To | Example Scope |
|---|---|---|---|
| Leave Policy | `<leave_policy>` | Benefits & Leave Administration | Annual leave, sick leave, parental, bereavement |
| Expense Claim | `<expense_claim>` | Finance & Accounts Payable | Business travel, meals, mileage, accommodations |
| Code of Conduct | `<code_of_conduct>` | Employee Relations & Legal Compliance | Harassment, anti-corruption, conflicts of interest |
| Remote Work | `<remote_work>` | Workplace Experience & People Operations | WFH guidelines, home equipment stipend, core hours |
| Health & Safety | `<health_safety>` | Facilities & HSE | Ergonomics, fire evacuation, incident reporting |

## App Features

- **Upload text files** (.txt, .md) for automatic paragraph-level classification
- **Direct text input** — paste policy text and classify instantly
- **Sample templates** — pre-loaded HR policy excerpts for quick testing
- **In-browser fine-tuning** — retrain the model directly from the sidebar
- **Confidence scores** — per-paragraph probability distribution across all categories
- **Category filtering** — show only specific categories via sidebar multiselect
- **Departmental routing** — each classification suggests the responsible department and email
- **Export** — download classified results as JSON or tagged text

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more HR policy documents and additional categories to improve model robustness.
- Implement a feature to display the model's confidence score for each classified section in the Streamlit app. ✅ Done
- Enhance the Streamlit interface for better user experience, perhaps allowing users to filter by category. ✅ Done
