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

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`.

## HR Categories & Departmental Routing

| Category | Tag | Routed To |
|---|---|---|
| Leave Policy | `<leave_policy>` | Benefits & Leave Administration |
| Expense Claim | `<expense_claim>` | Finance & Accounts Payable |
| Code of Conduct | `<code_of_conduct>` | Employee Relations & Legal Compliance |
| Remote Work | `<remote_work>` | Workplace Experience & People Operations |
| Health & Safety | `<health_safety>` | Facilities & HSE |

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
