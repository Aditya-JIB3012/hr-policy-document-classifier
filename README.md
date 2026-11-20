# HR Policy Document Classifier

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7cec7b2dmd/badge.svg)](https://ask.thili.ai/b/7cec7b2dmd)

The HR Policy Document Classifier is a tool designed for HR administrators to automate the classification of internal policy documents. By fine-tuning a GPT-2 model, it assigns categories like `<leave_policy>` or `<expense_claim>` to document sections, streamlining retrieval and departmental routing. This project showcases how a fine-tuned GPT-2 model can be used for practical text classification tasks in an HR context.

## What you are building

**Goal.** You will build a tool that automatically classifies HR policy document paragraphs into categories like 'leave policy' or 'expense claim', enabling quick retrieval and departmental routing for HR administrators. This project demonstrates the fine-tuning logic and inference pipeline for text classification.

**Done when.** The Streamlit app allows users to upload a text file, and it successfully displays classified sections of the document, with each paragraph correctly assigned to one of the predefined HR categories.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect example HR document paragraphs and assign 3-5 categories.**  
      Hint: Data Collection & Labeling: Gather diverse HR document paragraphs and manually assign appropriate categories.
- [ ] **2. Extend the GPT-2 tokenizer and model for these HR categories.**  
      Hint: Tokenizer/Model Extension: Modify the GPT-2 tokenizer and model to incorporate new special tokens for your HR categories, as shown in the `Fine-Tune GPT-2` drill.
- [ ] **3. Train the model using the prepared dataset and Hugging Face Trainer.**  
      Hint: Model Training: Use the Hugging Face `Trainer` class to fine-tune your extended GPT-2 model on the labeled HR dataset.
- [ ] **4. Implement a function to classify new document text.**  
      Hint: Inference Pipeline: Develop a Python function that takes new document text and uses your fine-tuned model to predict and generate category tags.
- [ ] **5. Create a Streamlit app to upload a text file and display classified sections.**  
      Hint: Application Development: Build a Streamlit application to provide a user interface for uploading documents and displaying the classified output.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more HR policy documents and additional categories to improve model robustness.
- Implement a feature to display the model's confidence score for each classified section in the Streamlit app.
- Enhance the Streamlit interface for better user experience, perhaps allowing users to filter by category.
