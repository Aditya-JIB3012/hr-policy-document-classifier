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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Train the model using the prepared dataset and Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement a function to classify new document text.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Create a Streamlit app to upload a text file and display classified sections.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

