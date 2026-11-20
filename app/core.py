"""Core ML and classification pipelines for the HR Policy Document Classifier.
One function per milestone, building an end-to-end GPT-2 fine-tuning and inference workflow.
"""
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 5 predefined HR Categories and their corresponding special tokens
CATEGORIES = [
    "leave_policy",
    "expense_claim",
    "code_of_conduct",
    "remote_work",
    "health_safety",
]

CATEGORY_TOKENS = [f"<{cat}>" for cat in CATEGORIES]

# Curated dataset of diverse HR document paragraphs
HR_EXAMPLES = [
    # leave_policy
    {
        "text": "All full-time employees are entitled to 20 days of paid annual leave per calendar year, accrued on a pro-rata monthly basis.",
        "category": "leave_policy",
    },
    {
        "text": "Sick leave certificates from a registered medical practitioner are mandatory for any consecutive absence exceeding two working days.",
        "category": "leave_policy",
    },
    {
        "text": "Eligible employees may take up to 16 weeks of paid parental leave upon the birth or adoption of a child, subject to 12 months of prior service.",
        "category": "leave_policy",
    },
    {
        "text": "Up to 5 unused annual leave days may be rolled over into the next financial year, provided they are utilized before March 31st.",
        "category": "leave_policy",
    },
    {
        "text": "Compassionate and bereavement leave grants up to 5 days of paid time off following the loss of an immediate family member.",
        "category": "leave_policy",
    },
    {
        "text": "Unpaid sabbatical leave for personal development or study may be requested after completing 3 continuous years of employment.",
        "category": "leave_policy",
    },

    # expense_claim
    {
        "text": "Business travel expenses must be submitted via the expense portal within 30 days of trip completion, accompanied by itemized tax receipts.",
        "category": "expense_claim",
    },
    {
        "text": "Meals incurred during client entertaining are reimbursable up to a per diem maximum of $75 per attendee with prior manager approval.",
        "category": "expense_claim",
    },
    {
        "text": "Mileage claims for business journeys made using personal vehicles will be compensated at the approved statutory rate of $0.65 per mile.",
        "category": "expense_claim",
    },
    {
        "text": "Economy class airfare is the default standard for domestic flights, while business class is only permissible for long-haul flights over 8 hours.",
        "category": "expense_claim",
    },
    {
        "text": "Subscriptions to professional associations and journals necessary for work roles can be claimed under departmental professional fees.",
        "category": "expense_claim",
    },
    {
        "text": "Hotel accommodation rates must adhere to preferred vendor corporate caps, excluding personal minibar or in-room movie charges.",
        "category": "expense_claim",
    },

    # code_of_conduct
    {
        "text": "Employees must treat all colleagues and clients with respect, maintaining a zero-tolerance stance against harassment, discrimination, and bullying.",
        "category": "code_of_conduct",
    },
    {
        "text": "Any actual or potential conflict of interest, including secondary employment or personal financial interests with suppliers, must be disclosed immediately.",
        "category": "code_of_conduct",
    },
    {
        "text": "Confidential company data, proprietary source code, and customer records must never be disclosed to unauthorized third parties or personal devices.",
        "category": "code_of_conduct",
    },
    {
        "text": "Acceptance of gifts, entertainment, or hospitality from vendors valued above $50 requires written declaration and clearance from compliance.",
        "category": "code_of_conduct",
    },
    {
        "text": "Whistleblower protections safeguard staff who report suspected unethical conduct, financial fraud, or regulatory violations in good faith.",
        "category": "code_of_conduct",
    },
    {
        "text": "Professional conduct is expected in public speaking and social media communications when representing or associating with the organization.",
        "category": "code_of_conduct",
    },

    # remote_work
    {
        "text": "Employees participating in hybrid arrangements may work remotely up to 3 days per week, subject to team synchronization and manager approval.",
        "category": "remote_work",
    },
    {
        "text": "Remote employees must maintain availability on designated team communication channels during core collaborative hours from 10:00 AM to 4:00 PM.",
        "category": "remote_work",
    },
    {
        "text": "A one-time home workstation ergonomics allowance of $500 is provided to acquire certified ergonomic desk furniture and monitors.",
        "category": "remote_work",
    },
    {
        "text": "All remote work sessions must connect through the corporate Virtual Private Network (VPN) with multi-factor authentication active at all times.",
        "category": "remote_work",
    },
    {
        "text": "Staff working from home must ensure a quiet, private workspace free from background distractions during sensitive customer video conferences.",
        "category": "remote_work",
    },
    {
        "text": "Relocating your primary remote work location across state or international borders requires advance tax and legal jurisdiction clearance.",
        "category": "remote_work",
    },

    # health_safety
    {
        "text": "All workplace injuries, hazardous spills, and near-miss incidents must be logged into the safety incident management portal within 24 hours.",
        "category": "health_safety",
    },
    {
        "text": "Building occupants must immediately evacuate via designated stairwells upon activation of fire alarm sounders and report to muster stations.",
        "category": "health_safety",
    },
    {
        "text": "Workstation ergonomic assessments may be scheduled through the health and safety team to mitigate repetitive strain injuries.",
        "category": "health_safety",
    },
    {
        "text": "First aid kits and automated external defibrillators (AEDs) are stationed on every office floor adjacent to the main kitchen and elevator banks.",
        "category": "health_safety",
    },
    {
        "text": "Smoking and the use of electronic cigarettes are strictly prohibited within company premises and within 10 meters of building entrances.",
        "category": "health_safety",
    },
    {
        "text": "Employees operating heavy equipment or handling laboratory chemicals must wear mandatory personal protective equipment (PPE) at all times.",
        "category": "health_safety",
    },
]


def format_example_for_gpt2(example: dict) -> str:
    """Format an HR text paragraph with its special category tag."""
    cat = example["category"]
    tag = f"<{cat}>" if not (cat.startswith("<") and cat.endswith(">")) else cat
    return f"{example['text'].strip()} {tag}"


# --- Milestone 1: Collect example HR document paragraphs and assign 3-5 categories.
# Hint: Data Collection & Labeling: Gather diverse HR document paragraphs and manually assign appropriate categories.
def milestone_1() -> dict:
    """Return labeled HR paragraphs and the set of 3-5 HR categories."""
    formatted_data = []
    for item in HR_EXAMPLES:
        formatted_data.append({
            "text": item["text"],
            "category": item["category"],
            "tag": f"<{item['category']}>",
            "formatted_text": format_example_for_gpt2(item),
        })
    return {
        "categories": list(CATEGORIES),
        "category_tokens": list(CATEGORY_TOKENS),
        "examples": formatted_data,
        "count": len(formatted_data),
    }


class ExtendedModelBundle:
    """Helper bundle holding extended tokenizer and model."""
    def __init__(self, tokenizer, model, special_tokens):
        self.tokenizer = tokenizer
        self.model = model
        self.special_tokens = special_tokens

    def __iter__(self):
        return iter((self.tokenizer, self.model))

    def __getitem__(self, item):
        if item == 0 or item == "tokenizer":
            return self.tokenizer
        elif item == 1 or item == "model":
            return self.model
        elif item == "special_tokens":
            return self.special_tokens
        raise KeyError(item)


def _load_pretrained(cls, model_name: str, **kwargs):
    """Load model/tokenizer, trying local cache first to avoid slow network timeouts."""
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, **kwargs)


def extend_tokenizer_and_model(model_name: str = "gpt2", categories: list[str] | None = None):
    """Extend GPT-2 tokenizer and model with special tokens for HR categories."""
    if categories is None:
        categories = CATEGORIES

    special_tokens = [f"<{cat}>" if not (cat.startswith("<") and cat.endswith(">")) else cat for cat in categories]

    tokenizer = _load_pretrained(GPT2Tokenizer, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = _load_pretrained(GPT2LMHeadModel, model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    return ExtendedModelBundle(tokenizer, model, special_tokens)


# --- Milestone 2: Extend the GPT-2 tokenizer and model for these HR categories.
# Hint: Tokenizer/Model Extension: Modify the GPT-2 tokenizer and model to incorporate new special tokens for your HR categories, as shown in the `Fine-Tune GPT-2` drill.
def milestone_2(model_name: str = "gpt2", categories: list[str] | None = None):
    """Extend the GPT-2 tokenizer and model with HR category special tokens."""
    return extend_tokenizer_and_model(model_name=model_name, categories=categories)



# --- Milestone 3: Train the model using the prepared dataset and Hugging Face Trainer.
# Hint: Model Training: Use the Hugging Face `Trainer` class to fine-tune your extended GPT-2 model on the labeled HR dataset.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Implement a function to classify new document text.
# Hint: Inference Pipeline: Develop a Python function that takes new document text and uses your fine-tuned model to predict and generate category tags.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Create a Streamlit app to upload a text file and display classified sections.
# Hint: Application Development: Build a Streamlit application to provide a user interface for uploading documents and displaying the classified output.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")

