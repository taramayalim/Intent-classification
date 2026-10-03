"""
app.py: Banking Intent Classification dashboard (Streamlit)

Pipeline: User Input -> Preprocessing -> Vectorizer -> Selected Model -> Prediction
Run with:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_text

st.set_page_config(page_title="Banking Intent Classifier", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS (styling only; no effect on the model or logic)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---------- Headings: bold ---------- */
h1 { font-size: 2.8rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { font-size: 2.1rem !important; font-weight: 800 !important; }
h3 { font-size: 1.7rem !important; font-weight: 700 !important; }
h4 { font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] h2 { font-size: 1.6rem !important; font-weight: 800 !important; }

/* ---------- Normal text: bigger ---------- */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    font-size: 1.2rem !important;
    line-height: 1.65 !important;
}
.stMarkdown strong { font-weight: 700 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    font-size: 1.1rem !important;
}
label, [data-testid="stWidgetLabel"] p {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}
textarea {
    font-size: 1.2rem !important;
    line-height: 1.6 !important;
}
code, pre, [data-testid="stCode"] * {
    font-size: 1.15rem !important;
}

/* ---------- Tabs: equal width, spaced, responsive ---------- */
[data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    width: 100% !important;
    gap: 0.75rem !important;
    padding-bottom: 0.25rem;
}
[data-testid="stTab"] {
    flex: 1 1 0 !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 0.9rem 1.2rem !important;
    border-radius: 12px 12px 0 0 !important;
    transition: background 0.2s ease;
}
[data-testid="stTab"]:hover {
    background: rgba(120, 120, 120, 0.10) !important;
}
[data-testid="stTab"], [data-testid="stTab"] *, button[role="tab"], button[role="tab"] * {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    white-space: normal !important;
}
/* Tablets and phones: tighter spacing, smaller labels */
@media (max-width: 768px) {
    [data-testid="stTabs"] [role="tablist"] { gap: 0.35rem !important; }
    [data-testid="stTab"] { padding: 0.7rem 0.4rem !important; }
    [data-testid="stTab"], [data-testid="stTab"] *, button[role="tab"], button[role="tab"] * {
        font-size: 1.05rem !important;
    }
}
@media (max-width: 480px) {
    [data-testid="stTab"], [data-testid="stTab"] *, button[role="tab"], button[role="tab"] * {
        font-size: 0.9rem !important;
    }
}

/* ---------- Buttons ---------- */
.stButton button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 0.55rem 0.9rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button, .stButton button * {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
.stButton button p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.35 !important;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18);
}
.stButton button[kind="primary"], .stButton button[kind="primary"] * {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}
.stButton button[kind="primary"] {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    padding: 0.7rem 2.2rem !important;
    background: linear-gradient(135deg, #1e6fd9, #7b3fe4) !important;
    color: #ffffff !important;
    border: none !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetricLabel"] p { font-size: 1.15rem !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { font-size: 2.4rem !important; font-weight: 800 !important; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] summary p { font-size: 1.2rem !important; font-weight: 700 !important; }

/* ---------- Important messages: creative alert cards ---------- */
@keyframes popIn {
    0%   { opacity: 0; transform: translateY(14px) scale(0.97); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes softPulse {
    0%, 100% { box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14); }
    50%      { box-shadow: 0 8px 30px rgba(0, 0, 0, 0.26); }
}
[data-testid="stAlertContainer"] {
    border-radius: 18px !important;
    border: none !important;
    border-left: 8px solid #888 !important;
    padding: 1.1rem 1.4rem !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
    animation: popIn 0.45s ease-out;
}
[data-testid="stAlertContainer"] p,
[data-testid="stAlertContainer"] * {
    font-size: 1.3rem !important;
    line-height: 1.55 !important;
}

/* Success = predicted intent (the key result) */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.28), rgba(39, 174, 96, 0.10)) !important;
    border-left-color: #27ae60 !important;
    animation: popIn 0.45s ease-out, softPulse 3s ease-in-out 0.5s infinite;
}
[data-testid="stAlertContentSuccess"] p,
[data-testid="stAlertContentSuccess"] * {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
}
[data-testid="stAlertContentSuccess"] p::before { content: "✅  "; }

/* Error = out-of-scope rejection */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.26), rgba(192, 57, 43, 0.08)) !important;
    border-left-color: #e74c3c !important;
}
[data-testid="stAlertContentError"] * { font-size: 1.45rem !important; font-weight: 700 !important; }

/* Warning = invalid or empty input */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    background: linear-gradient(135deg, rgba(243, 156, 18, 0.28), rgba(230, 126, 34, 0.08)) !important;
    border-left-color: #f39c12 !important;
}
[data-testid="stAlertContentWarning"] * { font-size: 1.4rem !important; font-weight: 700 !important; }

/* Info = low match strength / review note */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    background: linear-gradient(135deg, rgba(52, 152, 219, 0.26), rgba(41, 128, 185, 0.08)) !important;
    border-left-color: #3498db !important;
}
[data-testid="stAlertContentInfo"] * { font-size: 1.35rem !important; font-weight: 600 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Results from the notebook (update these if you retrain with new settings)
# ---------------------------------------------------------------------------
VALIDATION_RESULTS = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression", "Decision Tree", "SVM"],
    "Accuracy": [0.8242, 0.8384, 0.7194, 0.8490],
    "Precision (macro)": [0.8436, 0.8457, 0.7346, 0.8606],
    "Recall (macro)": [0.8249, 0.8456, 0.7262, 0.8561],
    "F1 (macro)": [0.8261, 0.8419, 0.7238, 0.8545],
}).set_index("Model")

TEST_ACCURACY = 0.8500
TEST_MACRO_F1 = 0.8498

EXAMPLES = [
    "My top up failed and the money was taken from my card",
    "How long does a transfer to another bank take?",
    "Why was I charged a fee on my transfer?",
    "I can't find the auto top up option",
    "The person I sent money to has not received it",
]

MAX_CHARS = 500

# Out-of-scope detection thresholds (chosen on the test set so ~95% of genuine
# queries pass while off-topic text is rejected; see README)
MIN_COVERAGE = 0.70    # share of cleaned words the vectorizer knows
MIN_SIMILARITY = 0.30  # cosine similarity to the closest training query


# ---------------------------------------------------------------------------
# Load saved artifacts once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.joblib")
    vectorizer = joblib.load("final_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    try:
        train_matrix = joblib.load("train_matrix.joblib")
    except FileNotFoundError:
        train_matrix = None  # app still works, using the vocabulary check only
    return model, vectorizer, label_encoder, train_matrix


try:
    model, vectorizer, label_encoder, train_matrix = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Model file not found: {e.filename}. Place final_model.joblib, "
        "final_vectorizer.joblib and label_encoder.joblib in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pretty(intent):
    return intent.replace("_", " ").title()


def validate(text):
    """Return an error message for invalid input, or None if the input is fine."""
    if not text or not text.strip():
        return "Please enter a customer message."
    if len(text) > MAX_CHARS:
        return f"Message is too long ({len(text)} characters). Please keep it under {MAX_CHARS}."
    if not any(ch.isalpha() for ch in text):
        return "The message contains no letters. Please enter a sentence in English."
    return None


def predict(text):
    """
    Full pipeline with an out-of-scope check.
    Returns a dict with status 'ok', 'empty' or 'out_of_scope'.
    """
    cleaned = clean_text(text)                          # 1. same preprocessing as training
    if not cleaned:
        return {"status": "empty"}

    tokens = cleaned.split()
    vocab = vectorizer.vocabulary_
    coverage = float(np.mean([t in vocab for t in tokens]))

    features = vectorizer.transform([cleaned])          # 2. trained TF-IDF vectorizer
    similarity = None
    if train_matrix is not None and features.nnz > 0:
        similarity = float(cosine_similarity(features, train_matrix).max())

    info = {"cleaned": cleaned, "coverage": coverage, "similarity": similarity}

    # Out-of-scope check: the model can only choose among its 21 intents, so
    # off-topic text must be rejected BEFORE prediction.
    if features.nnz == 0 or coverage < MIN_COVERAGE or (
        similarity is not None and similarity < MIN_SIMILARITY
    ):
        return {"status": "out_of_scope", **info}

    scores = model.decision_function(features)[0]       # 3. trained SVM
    order = np.argsort(scores)[::-1][:3]
    intents = label_encoder.inverse_transform(model.classes_[order])
    return {"status": "ok", "intents": intents, **info}


def match_strength(coverage, similarity):
    """How closely the message resembles known queries (heuristic, not a probability)."""
    if similarity is None:
        return ("High", "🟢") if coverage == 1.0 else ("Medium", "🟡")
    if coverage == 1.0 and similarity >= 0.60:
        return "High", "🟢"
    if similarity >= 0.45:
        return "Medium", "🟡"
    return "Low", "🔴"


def set_example(text):
    st.session_state["query"] = text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this system")
    st.write(
        "Classifies customer messages about **transfers and top-ups** into one of "
        f"**{len(label_encoder.classes_)} intents** so the Payments & Balance team "
        "can route them automatically."
    )
    st.markdown(
        "**Pipeline**\n\n"
        "1. User input\n2. Preprocessing\n3. TF-IDF (unigrams + bigrams)\n"
        "4. Linear SVM\n5. Predicted intent"
    )
    st.divider()
    st.metric("Test accuracy", f"{TEST_ACCURACY:.2%}")
    st.metric("Test macro F1", f"{TEST_MACRO_F1:.4f}")
    st.caption("Dataset: BANKING77 (money movement subset). Classical NLP only.")
    if train_matrix is None:
        st.warning("train_matrix.joblib not found. Using the vocabulary check only; "
                   "off-topic detection is weaker.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🏦 Banking Intent Classifier")
st.caption("Money movement queries: transfers and top-ups")

tab_predict, tab_perf, tab_intents = st.tabs(["Predict", "Model performance", "Supported intents"])

# ----- Predict tab -----------------------------------------------------------
with tab_predict:
    st.subheader("Enter a customer message")

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(ex[:28] + "…", key=ex, on_click=set_example, args=(ex,), use_container_width=True)

    user_text = st.text_area("Customer message", key="query", height=110,
                             placeholder="e.g. My transfer has been pending for three days")

    if st.button("Classify", type="primary"):
        error = validate(user_text)
        if error:
            st.warning(error)
        else:
            result = predict(user_text)
            if result["status"] == "empty":
                st.warning(
                    "After cleaning, no meaningful words were left (the message contained "
                    "only common words). Please describe the problem in more detail."
                )
            elif result["status"] == "out_of_scope":
                st.error(
                    "🚫 This message does not look like a money transfer or top-up query, "
                    "so no prediction is made. Please describe a transfer or top-up problem."
                )
                with st.expander("Why was this rejected?"):
                    st.write(f"Text after preprocessing: `{result.get('cleaned', '')}`")
                    st.write(f"Known-word coverage: **{result.get('coverage', 0):.0%}** "
                             f"(minimum {MIN_COVERAGE:.0%})")
                    if result.get("similarity") is not None:
                        st.write(f"Similarity to closest training query: "
                                 f"**{result['similarity']:.2f}** (minimum {MIN_SIMILARITY:.2f})")
            else:
                level, icon = match_strength(result["coverage"], result["similarity"])
                left, right = st.columns([1, 1])

                with left:
                    st.markdown("#### Predicted intent")
                    st.success(f"**{pretty(result['intents'][0])}**")
                    st.write(f"Match strength: {icon} **{level}**")
                    if level == "Low":
                        st.info("This message only weakly resembles known queries. "
                                "It may need manual review.")
                    st.markdown("**Text after preprocessing**")
                    st.code(result["cleaned"], language=None)

                with right:
                    st.markdown("#### Top 3 candidate intents (ranked)")
                    for rank, name in enumerate(result["intents"], start=1):
                        st.write(f"{rank}. {pretty(name)}")
                    st.markdown("**Input checks**")
                    st.write(f"Known-word coverage: **{result['coverage']:.0%}**")
                    if result["similarity"] is not None:
                        st.write(f"Similarity to closest training query: "
                                 f"**{result['similarity']:.2f}**")
                    st.caption(
                        "Match strength reflects how similar the message is to the training "
                        "data. It is a heuristic, not a probability."
                    )

# ----- Performance tab -------------------------------------------------------
with tab_perf:
    st.subheader("Model comparison (validation set, tuned models)")
    st.dataframe(VALIDATION_RESULTS.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 use_container_width=True)
    st.bar_chart(VALIDATION_RESULTS[["Accuracy", "F1 (macro)"]])

    c1, c2 = st.columns(2)
    c1.metric("Final test accuracy (SVM)", f"{TEST_ACCURACY:.4f}")
    c2.metric("Final test macro F1 (SVM)", f"{TEST_MACRO_F1:.4f}")

    st.markdown(
        "**Why SVM was selected:** it had the highest macro F1 on validation data. "
        "Macro F1 averages the F1-score over all 21 intents equally, so it reflects "
        "performance on smaller intents as well as large ones, which is why it is used "
        "instead of accuracy alone."
    )

# ----- Intents tab -----------------------------------------------------------
with tab_intents:
    st.subheader("Intents the model can predict")
    names = sorted(label_encoder.classes_)
    transfers = [pretty(n) for n in names if "top_up" not in n and "topping_up" not in n]
    topups = [pretty(n) for n in names if "top_up" in n or "topping_up" in n]
    a, b = st.columns(2)
    a.markdown("**Transfers**")
    a.write("\n".join(f"- {n}" for n in transfers))
    b.markdown("**Top-ups**")
    b.write("\n".join(f"- {n}" for n in topups))
