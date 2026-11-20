# Evaluation Results

## Model
- **Base model:** GPT-2 (124M parameters)
- **Extended with:** 5 special HR category tokens
- **Training data:** 30 curated paragraphs across 5 categories
- **Test data:** 15 held-out paragraphs (never seen during training)
- **Training:** 3 epochs, batch size 4, learning rate 5e-5

## Results Summary

| Metric | Value |
|--------|-------|
| Training Loss | 3.9963 |
| Baseline (pre-training) Test Accuracy | 26.7% |
| Training Set Accuracy (post-training) | 36.7% |
| **Test Set Accuracy (post-training)** | **33.3%** |

## Per-Category Test Accuracy

| Category | Accuracy | Correct / Total |
|----------|----------|-----------------|
| leave_policy | 100.0% | 3/3 |
| expense_claim | 66.7% | 2/3 |
| code_of_conduct | 0.0% | 0/3 |
| remote_work | 0.0% | 0/3 |
| health_safety | 0.0% | 0/3 |

## Sample Test Predictions

| Status | Text (truncated) | Actual | Predicted | Confidence |
|--------|-----------------|--------|-----------|------------|
| ✅ | Employees may request up to 10 days of unpaid leave annually... | leave_policy | leave_policy | 0.203 |
| ✅ | Maternity and paternity leave policies comply with federal a... | leave_policy | leave_policy | 0.203 |
| ✅ | Jury duty leave is granted with full pay for the duration of... | leave_policy | leave_policy | 0.203 |
| ✅ | Conference registration fees for approved industry events ar... | expense_claim | expense_claim | 0.202 |
| ✅ | Taxi and rideshare expenses for client meetings are reimburs... | expense_claim | expense_claim | 0.203 |
| ❌ | Employees may claim up to $200 per month for mobile phone co... | expense_claim | leave_policy | 0.203 |
| ❌ | Employees are prohibited from using company resources, inclu... | code_of_conduct | leave_policy | 0.202 |
| ❌ | All staff must complete mandatory anti-corruption and briber... | code_of_conduct | leave_policy | 0.202 |
| ❌ | Insider trading based on non-public company information is s... | code_of_conduct | leave_policy | 0.202 |
| ❌ | Remote employees must complete a home office safety checklis... | remote_work | expense_claim | 0.202 |
| ❌ | Video conferencing with cameras on is mandatory during all-h... | remote_work | expense_claim | 0.202 |
| ❌ | International remote work requests require prior approval fr... | remote_work | leave_policy | 0.203 |
| ❌ | All new hires must complete fire safety orientation within t... | health_safety | leave_policy | 0.202 |
| ❌ | Hazardous material spill response kits are located in every ... | health_safety | expense_claim | 0.202 |
| ❌ | Mental health support services including counseling and an e... | health_safety | leave_policy | 0.203 |

## Limitations & Honest Assessment

- **Small training set:** Only 30 examples (6 per category). Real-world deployment would need hundreds or thousands.
- **GPT-2 is a generative model, not a classifier by design.** We repurpose its next-token prediction for classification by comparing logits over special tokens. A purpose-built classifier (e.g., BERT with a classification head) would likely outperform this approach.
- **No real generalization guarantee:** The model memorizes patterns from limited examples. Accuracy on truly novel policy language may be lower.
- **Category coverage is narrow:** 5 categories cover common HR topics but miss areas like compensation, performance review, training & development, etc.
- **No confidence calibration:** Reported confidence scores are raw softmax probabilities over category tokens and may not reflect true prediction reliability.
- **CPU-only training:** Training on CPU is slow for larger datasets. GPU acceleration would be needed for production scale.

## Potential Improvements

- Expand the dataset to 100+ examples per category with real HR policy documents
- Use a classification-oriented model (BERT, DeBERTa) instead of GPT-2
- Add data augmentation (paraphrasing, synonym replacement)
- Implement cross-validation instead of a single train/test split
- Add confidence calibration (temperature scaling)
- Include more HR categories (compensation, performance, training, etc.)

*Generated automatically by `evaluate.py`*