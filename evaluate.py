"""Evaluate the HR Policy Document Classifier.

Runs the fine-tuned model against held-out test examples and training data,
computes accuracy, per-category precision, and saves results to results/.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core import (
    CATEGORIES,
    HR_EXAMPLES,
    extend_tokenizer_and_model,
    train_hr_model,
    classify_paragraph,
    classify_document,
)


# Held-out test examples — NOT in the training set
TEST_EXAMPLES = [
    # leave_policy
    {"text": "Employees may request up to 10 days of unpaid leave annually for personal matters, subject to departmental approval.", "category": "leave_policy"},
    {"text": "Maternity and paternity leave policies comply with federal and state regulations, granting eligible employees job-protected time off.", "category": "leave_policy"},
    {"text": "Jury duty leave is granted with full pay for the duration of service, and employees must provide court documentation.", "category": "leave_policy"},

    # expense_claim
    {"text": "Conference registration fees for approved industry events are fully reimbursable when pre-authorized by the department head.", "category": "expense_claim"},
    {"text": "Taxi and rideshare expenses for client meetings are reimbursable with digital receipts submitted through the online portal.", "category": "expense_claim"},
    {"text": "Employees may claim up to $200 per month for mobile phone costs when their role requires frequent client communication.", "category": "expense_claim"},

    # code_of_conduct
    {"text": "Employees are prohibited from using company resources, including email and internet access, for personal political campaigning.", "category": "code_of_conduct"},
    {"text": "All staff must complete mandatory anti-corruption and bribery training annually as part of regulatory compliance requirements.", "category": "code_of_conduct"},
    {"text": "Insider trading based on non-public company information is strictly prohibited and may result in criminal prosecution.", "category": "code_of_conduct"},

    # remote_work
    {"text": "Remote employees must complete a home office safety checklist and submit photographic evidence of their workspace setup.", "category": "remote_work"},
    {"text": "Video conferencing with cameras on is mandatory during all-hands meetings and client presentations for remote team members.", "category": "remote_work"},
    {"text": "International remote work requests require prior approval from HR and legal departments due to tax and employment law implications.", "category": "remote_work"},

    # health_safety
    {"text": "All new hires must complete fire safety orientation within their first week, including hands-on extinguisher training.", "category": "health_safety"},
    {"text": "Hazardous material spill response kits are located in every laboratory and must be inspected monthly by designated safety officers.", "category": "health_safety"},
    {"text": "Mental health support services including counseling and an employee assistance program are available free and confidentially to all staff.", "category": "health_safety"},
]


def evaluate_model(model, tokenizer, examples, label=""):
    """Evaluate model on a set of labeled examples. Returns metrics dict."""
    correct = 0
    total = len(examples)
    per_category = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    predictions = []

    for ex in examples:
        result = classify_paragraph(
            ex["text"],
            model=model,
            tokenizer=tokenizer,
        )
        predicted = result["category"]
        actual = ex["category"]
        is_correct = predicted == actual

        if is_correct:
            correct += 1
        per_category[actual]["total"] += 1
        if is_correct:
            per_category[actual]["correct"] += 1

        pred_record = {
            "text": ex["text"][:100] + "..." if len(ex["text"]) > 100 else ex["text"],
            "actual": actual,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": result["confidence"],
            "scores": result["scores"],
        }
        per_category[actual]["predictions"].append(pred_record)
        predictions.append(pred_record)

    accuracy = correct / total if total > 0 else 0.0

    category_metrics = {}
    for cat in CATEGORIES:
        cat_data = per_category[cat]
        cat_total = cat_data["total"]
        cat_correct = cat_data["correct"]
        category_metrics[cat] = {
            "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
            "correct": cat_correct,
            "total": cat_total,
        }

    return {
        "label": label,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_category": category_metrics,
        "predictions": predictions,
    }


def main():
    print("=" * 70)
    print("HR Policy Document Classifier — Evaluation Report")
    print("=" * 70)

    # Step 1: Load extended model (untrained baseline)
    print("\n[1/4] Loading extended GPT-2 model (pre-fine-tune baseline)...")
    bundle = extend_tokenizer_and_model()
    tokenizer, model = bundle.tokenizer, bundle.model

    baseline_test = evaluate_model(model, tokenizer, TEST_EXAMPLES, label="Baseline (Pre-Training) on Test Set")
    print(f"  Baseline Test Accuracy: {baseline_test['overall_accuracy']*100:.1f}% ({baseline_test['correct']}/{baseline_test['total']})")

    # Step 2: Fine-tune the model
    print("\n[2/4] Fine-tuning GPT-2 on 30 HR training examples (3 epochs)...")
    train_result = train_hr_model(
        model=model,
        tokenizer=tokenizer,
        epochs=3,
        batch_size=4,
    )
    trained_model = train_result["model"]
    final_loss = train_result["train_loss"]
    print(f"  Training complete. Final loss: {final_loss:.4f}")

    # Step 3: Evaluate on training set (sanity check)
    print("\n[3/4] Evaluating on training set (sanity check)...")
    train_eval = evaluate_model(trained_model, tokenizer, HR_EXAMPLES, label="Post-Training on Training Set")
    print(f"  Training Set Accuracy: {train_eval['overall_accuracy']*100:.1f}% ({train_eval['correct']}/{train_eval['total']})")

    # Step 4: Evaluate on held-out test set
    print("\n[4/4] Evaluating on held-out test set (15 unseen examples)...")
    test_eval = evaluate_model(trained_model, tokenizer, TEST_EXAMPLES, label="Post-Training on Test Set")
    print(f"  Test Set Accuracy: {test_eval['overall_accuracy']*100:.1f}% ({test_eval['correct']}/{test_eval['total']})")

    # Print per-category breakdown
    print("\n" + "-" * 70)
    print("Per-Category Test Set Accuracy:")
    print("-" * 70)
    for cat in CATEGORIES:
        m = test_eval["per_category"][cat]
        bar = "█" * int(m["accuracy"] * 20) + "░" * (20 - int(m["accuracy"] * 20))
        print(f"  {cat:<20s} {bar} {m['accuracy']*100:5.1f}%  ({m['correct']}/{m['total']})")

    # Print sample predictions
    print("\n" + "-" * 70)
    print("Sample Predictions (Test Set):")
    print("-" * 70)
    for p in test_eval["predictions"]:
        status = "✓" if p["correct"] else "✗"
        print(f"  {status} [{p['actual']:<18s}→ {p['predicted']:<18s}] conf={p['confidence']:.3f}  {p['text'][:80]}")

    # Save results
    os.makedirs("results", exist_ok=True)

    full_report = {
        "model": "gpt2 (124M params) + 5 special tokens",
        "training_data": f"{len(HR_EXAMPLES)} HR paragraphs across {len(CATEGORIES)} categories",
        "test_data": f"{len(TEST_EXAMPLES)} held-out HR paragraphs (not in training set)",
        "training_epochs": 3,
        "training_loss": round(final_loss, 4),
        "baseline_test_accuracy": baseline_test["overall_accuracy"],
        "training_set_accuracy": train_eval["overall_accuracy"],
        "test_set_accuracy": test_eval["overall_accuracy"],
        "per_category_test_accuracy": test_eval["per_category"],
        "baseline_predictions": baseline_test["predictions"],
        "test_predictions": test_eval["predictions"],
    }

    with open("results/evaluation_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    # Generate a human-readable summary
    summary_lines = [
        "# Evaluation Results",
        "",
        "## Model",
        "- **Base model:** GPT-2 (124M parameters)",
        "- **Extended with:** 5 special HR category tokens",
        f"- **Training data:** {len(HR_EXAMPLES)} curated paragraphs across {len(CATEGORIES)} categories",
        f"- **Test data:** {len(TEST_EXAMPLES)} held-out paragraphs (never seen during training)",
        f"- **Training:** 3 epochs, batch size 4, learning rate 5e-5",
        "",
        "## Results Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Training Loss | {final_loss:.4f} |",
        f"| Baseline (pre-training) Test Accuracy | {baseline_test['overall_accuracy']*100:.1f}% |",
        f"| Training Set Accuracy (post-training) | {train_eval['overall_accuracy']*100:.1f}% |",
        f"| **Test Set Accuracy (post-training)** | **{test_eval['overall_accuracy']*100:.1f}%** |",
        "",
        "## Per-Category Test Accuracy",
        "",
        "| Category | Accuracy | Correct / Total |",
        "|----------|----------|-----------------|",
    ]

    for cat in CATEGORIES:
        m = test_eval["per_category"][cat]
        summary_lines.append(f"| {cat} | {m['accuracy']*100:.1f}% | {m['correct']}/{m['total']} |")

    summary_lines.extend([
        "",
        "## Sample Test Predictions",
        "",
        "| Status | Text (truncated) | Actual | Predicted | Confidence |",
        "|--------|-----------------|--------|-----------|------------|",
    ])
    for p in test_eval["predictions"]:
        status = "✅" if p["correct"] else "❌"
        txt = p["text"][:60].replace("|", "\\|") + "..."
        summary_lines.append(f"| {status} | {txt} | {p['actual']} | {p['predicted']} | {p['confidence']:.3f} |")

    summary_lines.extend([
        "",
        "## Limitations & Honest Assessment",
        "",
        "- **Small training set:** Only 30 examples (6 per category). Real-world deployment would need hundreds or thousands.",
        "- **GPT-2 is a generative model, not a classifier by design.** We repurpose its next-token prediction for classification by comparing logits over special tokens. A purpose-built classifier (e.g., BERT with a classification head) would likely outperform this approach.",
        "- **No real generalization guarantee:** The model memorizes patterns from limited examples. Accuracy on truly novel policy language may be lower.",
        "- **Category coverage is narrow:** 5 categories cover common HR topics but miss areas like compensation, performance review, training & development, etc.",
        "- **No confidence calibration:** Reported confidence scores are raw softmax probabilities over category tokens and may not reflect true prediction reliability.",
        "- **CPU-only training:** Training on CPU is slow for larger datasets. GPU acceleration would be needed for production scale.",
        "",
        "## Potential Improvements",
        "",
        "- Expand the dataset to 100+ examples per category with real HR policy documents",
        "- Use a classification-oriented model (BERT, DeBERTa) instead of GPT-2",
        "- Add data augmentation (paraphrasing, synonym replacement)",
        "- Implement cross-validation instead of a single train/test split",
        "- Add confidence calibration (temperature scaling)",
        "- Include more HR categories (compensation, performance, training, etc.)",
        "",
        f"*Generated automatically by `evaluate.py`*",
    ])

    with open("results/evaluation_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n✓ Results saved to results/evaluation_report.json")
    print(f"✓ Summary saved to results/evaluation_results.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
