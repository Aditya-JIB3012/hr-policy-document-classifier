"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Collect example HR document paragraphs and assign 3-5 categories.
def test_milestone_1():
    from app.core import milestone_1, CATEGORIES
    data = milestone_1()
    assert isinstance(data, dict)
    assert "categories" in data
    assert "examples" in data
    assert 3 <= len(data["categories"]) <= 5
    assert len(data["examples"]) >= 15
    for ex in data["examples"]:
        assert "text" in ex and len(ex["text"]) > 10
        assert "category" in ex and ex["category"] in CATEGORIES
        assert "tag" in ex and ex["tag"] == f"<{ex['category']}>"
        assert ex["category"] in ex["formatted_text"]



# Milestone 2: Extend the GPT-2 tokenizer and model for these HR categories.
def test_milestone_2():
    from app.core import milestone_2, CATEGORIES, CATEGORY_TOKENS
    bundle = milestone_2()
    tokenizer, model = bundle

    # Verify special tokens are present in tokenizer
    for token in CATEGORY_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None
        assert token_id != tokenizer.unk_token_id

    # Verify tokenizer vocab size has grown by category tokens
    assert len(tokenizer) == 50257 + len(CATEGORIES)

    # Verify model embedding matrix was resized to match tokenizer vocab
    embedding_weight = model.get_input_embeddings().weight
    assert embedding_weight.shape[0] == len(tokenizer)



# Milestone 3: Train the model using the prepared dataset and Hugging Face Trainer.
def test_milestone_3(tmp_path):
    from transformers import Trainer
    from app.core import milestone_2, milestone_3

    bundle = milestone_2()
    # Train using Trainer for 2 steps in temporary directory to verify pipeline
    result = milestone_3(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        max_steps=2,
        output_dir=str(tmp_path / "trainer_ckpt"),
    )

    assert isinstance(result, dict)
    assert "trainer" in result
    assert isinstance(result["trainer"], Trainer)
    assert "train_loss" in result
    assert result["train_loss"] is not None
    assert result["train_loss"] > 0



# Milestone 4: Implement a function to classify new document text.
def test_milestone_4():
    from app.core import milestone_2, milestone_4, classify_document, CATEGORIES

    bundle = milestone_2()
    tokenizer, model = bundle

    # Single paragraph inference
    sample_text = "Sick leave certificates are required for absences extending past three consecutive days."
    res = milestone_4(sample_text, model=model, tokenizer=tokenizer)
    assert isinstance(res, dict)
    assert res["category"] in CATEGORIES
    assert res["tag"] == f"<{res['category']}>"
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["scores"]) == len(CATEGORIES)
    assert sample_text in res["tagged_text"]

    # Multi-paragraph document inference
    multi_doc = (
        "All employees are entitled to 20 days paid leave per year.\n\n"
        "Business mileage is reimbursed at $0.60 per mile driven."
    )
    doc_results = classify_document(multi_doc, model=model, tokenizer=tokenizer)
    assert isinstance(doc_results, list)
    assert len(doc_results) == 2
    for item in doc_results:
        assert item["category"] in CATEGORIES
        assert "confidence" in item



# Milestone 5: Create a Streamlit app to upload a text file and display classified sections.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

