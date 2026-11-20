"""Streamlit entry point: `streamlit run app/main.py`
HR Policy Document Classifier with GPT-2 fine-tuned classification and routing.
"""
import io
import json
import streamlit as st
from app.core import (
    CATEGORIES,
    CATEGORY_TOKENS,
    extend_tokenizer_and_model,
    classify_document,
    train_hr_model,
    milestone_1,
)

st.set_page_config(
    page_title="HR Policy Document Classifier",
    page_icon="📋",
    layout="wide",
)

# Departmental routing recommendations based on category
DEPARTMENT_ROUTING = {
    "leave_policy": {
        "dept": "Benefits & Leave Administration",
        "email": "hr-leave@company.internal",
        "color": "#10b981",  # Emerald
        "badge": "🟢 Leave Policy",
    },
    "expense_claim": {
        "dept": "Finance & Accounts Payable",
        "email": "expenses@company.internal",
        "color": "#f59e0b",  # Amber
        "badge": "🟡 Expense Claim",
    },
    "code_of_conduct": {
        "dept": "Employee Relations & Legal Compliance",
        "email": "compliance@company.internal",
        "color": "#3b82f6",  # Blue
        "badge": "🔵 Code of Conduct",
    },
    "remote_work": {
        "dept": "Workplace Experience & People Operations",
        "email": "workplace@company.internal",
        "color": "#8b5cf6",  # Purple
        "badge": "🟣 Remote Work",
    },
    "health_safety": {
        "dept": "Facilities & HSE (Health, Safety & Environment)",
        "email": "safety@company.internal",
        "color": "#ef4444",  # Red
        "badge": "🔴 Health & Safety",
    },
}

SAMPLE_DOCUMENTS = {
    "Employee Handbook Excerpt": (
        "All permanent full-time staff are eligible for 20 working days of annual leave each year, accrued monthly.\n\n"
        "Business travel expenses including ground transportation and meals up to $75 per diem must be filed with itemized receipts.\n\n"
        "Employees must adhere strictly to conflict of interest guidelines and refrain from accepting gifts exceeding $50 from prospective vendors.\n\n"
        "Remote team members are required to log into the corporate VPN and maintain active presence during core collaboration hours from 10:00 AM to 4:00 PM.\n\n"
        "Emergency evacuation muster drills occur quarterly; all employees must follow designated stairwell routes immediately when sirens sound."
    ),
    "Remote & Travel Policy Brief": (
        "Eligible staff may work from home up to three days per week after completing the probationary period and receiving manager approval.\n\n"
        "Flight bookings for business travel exceeding 5 hours must be arranged through corporate travel agents in economy plus class.\n\n"
        "Workstation ergonomic equipment allowances of $500 are available for all permanent remote employees upon request."
    ),
    "Compliance & Leave Notice": (
        "Medical certificates from registered physicians are required for any sick leave absence exceeding two consecutive business days.\n\n"
        "The company maintains a zero-tolerance stance against workplace bullying, harassment, and discrimination of any nature."
    ),
}


@st.cache_resource(show_spinner="Loading fine-tuned GPT-2 tokenizer and model...")
def get_cached_model():
    """Load and cache the extended GPT-2 model in memory."""
    bundle = extend_tokenizer_and_model()
    return bundle.tokenizer, bundle.model


def main():
    st.title("📋 HR Policy Document Classifier")
    st.markdown(
        "HR administrators use this tool to automatically classify internal policy documents "
        "(e.g., **leave policy**, **expense claim**, **code of conduct**) for quick retrieval and departmental routing."
    )

    tokenizer, model = get_cached_model()

    # Sidebar: Model configuration and filtering
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.info("Base Model: **GPT-2** (Causal LM)")
        st.write("**Special Category Tokens:**")
        for tok in CATEGORY_TOKENS:
            st.code(tok, language="text")

        st.markdown("---")
        st.header("🔍 Filter Results")
        selected_categories = st.multiselect(
            "Filter by Category",
            options=CATEGORIES,
            default=CATEGORIES,
            help="Show only paragraphs classified into selected categories.",
        )
        min_confidence = st.slider(
            "Minimum Confidence Filter (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        ) / 100.0

        st.markdown("---")
        st.subheader("🎓 In-Browser Training")
        st.caption("Fine-tune the model directly on the curated HR training dataset.")
        if st.button("Run Quick Fine-Tune (1 Epoch)"):
            with st.spinner("Fine-tuning extended GPT-2 with Hugging Face Trainer..."):
                train_res = train_hr_model(
                    model=model,
                    tokenizer=tokenizer,
                    epochs=1,
                    batch_size=4,
                )
                st.success(f"Trained successfully! Loss: {train_res['train_loss']:.4f}")

    # Tabs for inputs
    tab_upload, tab_paste, tab_samples, tab_info = st.tabs([
        "📁 Upload Document",
        "✍️ Direct Text Input",
        "📑 Sample Templates",
        "ℹ️ About & Categories",
    ])

    document_text = ""

    with tab_upload:
        st.subheader("Upload an HR Policy Document")
        uploaded_file = st.file_uploader(
            "Upload a plain text or markdown document (.txt, .md)",
            type=["txt", "md"],
            help="The document will be split into paragraphs and classified automatically.",
        )
        if uploaded_file is not None:
            try:
                document_text = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded '{uploaded_file.name}' ({len(document_text)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab_paste:
        st.subheader("Paste Document Content")
        pasted_text = st.text_area(
            "Enter policy paragraphs (separate paragraphs with blank lines):",
            value="",
            height=200,
            placeholder="Paste your HR document paragraphs here...",
        )
        if pasted_text.strip():
            document_text = pasted_text

    with tab_samples:
        st.subheader("Pre-loaded HR Policy Samples")
        sample_choice = st.selectbox(
            "Choose a sample policy document to test:",
            options=list(SAMPLE_DOCUMENTS.keys()),
        )
        if st.button("Load Selected Sample"):
            st.session_state["active_sample"] = SAMPLE_DOCUMENTS[sample_choice]

        if "active_sample" in st.session_state:
            document_text = st.session_state["active_sample"]
            st.text_area("Selected Sample Text:", value=document_text, height=180, disabled=True)

    with tab_info:
        st.subheader("Predefined HR Categories & Drill Architecture")
        m1_data = milestone_1()
        st.write(f"Total training examples curated: **{m1_data['count']}** across **{len(m1_data['categories'])}** categories:")
        for cat in CATEGORIES:
            info = DEPARTMENT_ROUTING[cat]
            st.markdown(f"- **`{cat}`** ({info['badge']}) → Routing: *{info['dept']}*")

    st.markdown("---")

    # Document Classification Action
    if document_text.strip():
        st.subheader("Document Preview")
        with st.expander("Show Raw Document Text", expanded=False):
            st.text(document_text)

        if st.button("🚀 Classify Document Paragraphs", type="primary"):
            with st.spinner("Analyzing document and classifying paragraphs..."):
                results = classify_document(
                    document_text=document_text,
                    model=model,
                    tokenizer=tokenizer,
                )
                st.session_state["classification_results"] = results

    # Render Results
    if "classification_results" in st.session_state:
        results = st.session_state["classification_results"]
        st.markdown("## 📊 Classification & Routing Output")

        if not results:
            st.warning("No valid paragraphs found in the document.")
            return

        # Overview Metrics
        total_paras = len(results)
        cat_counts = {}
        for r in results:
            cat = r["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        top_cat = max(cat_counts, key=cat_counts.get)
        avg_conf = sum(r["confidence"] for r in results) / total_paras

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Paragraphs", total_paras)
        with col2:
            st.metric("Dominant Category", top_cat.replace("_", " ").title())
        with col3:
            st.metric("Average Confidence", f"{avg_conf * 100:.1f}%")

        # Category Distribution Pills
        st.write("**Identified Categories:**")
        cols = st.columns(len(cat_counts))
        for col, (cat, count) in zip(cols, cat_counts.items()):
            routing = DEPARTMENT_ROUTING.get(cat, {"badge": cat, "color": "#6b7280"})
            with col:
                st.markdown(
                    f"<div style='background:{routing['color']}22; border-left: 4px solid {routing['color']}; "
                    f"padding: 8px; border-radius: 4px;'><strong>{routing['badge']}</strong>: {count} section(s)</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("### Classified Sections")

        filtered_results = [
            r for r in results
            if r["category"] in selected_categories and r["confidence"] >= min_confidence
        ]

        if not filtered_results:
            st.info("No paragraphs match the current category and confidence filters.")
        else:
            for idx, item in enumerate(filtered_results, start=1):
                cat = item["category"]
                routing = DEPARTMENT_ROUTING.get(cat, {"dept": "HR Operations", "email": "hr@company.internal", "color": "#333", "badge": cat})
                conf_pct = item["confidence"] * 100

                with st.container():
                    st.markdown(
                        f"<div style='border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #fafafa;'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;'>"
                        f"<span style='font-weight:bold; font-size: 1.1em; color:{routing['color']};'>{routing['badge']} (Paragraph {idx})</span>"
                        f"<span style='background:{routing['color']}22; color:{routing['color']}; font-weight:600; padding:2px 8px; border-radius:12px;'>"
                        f"Confidence: {conf_pct:.1f}%</span>"
                        f"</div>"
                        f"<p style='font-size:1.0em; line-height:1.5; color:#1e293b;'>{item['paragraph']}</p>"
                        f"<div style='font-size:0.85em; color:#475569; margin-top:8px;'>"
                        f"📬 <strong>Departmental Route:</strong> {routing['dept']} (<code>{routing['email']}</code>)"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"View Probabilities & Tag for Paragraph {idx}"):
                        st.code(item["tagged_text"], language="text")
                        for sc_cat, sc_val in sorted(item["scores"].items(), key=lambda x: x[1], reverse=True):
                            st.write(f"- `{sc_cat}`: {sc_val * 100:.2f}%")

        # Export & Download Section
        st.markdown("### 💾 Export Results")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_str,
                file_name="classified_hr_document.json",
                mime="application/json",
            )

        with export_col2:
            tagged_document = "\n\n".join([r["tagged_text"] for r in results])
            st.download_button(
                label="📥 Download Tagged Text Document",
                data=tagged_document,
                file_name="tagged_hr_document.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
