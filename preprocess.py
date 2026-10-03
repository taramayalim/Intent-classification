"""
preprocess.py
Text cleaning used for BOTH training (notebook) and deployment (app.py).
This is the exact clean_text() from the notebook. If you change it, you must
re-run the notebook, retrain, and re-save the model and vectorizer.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words where 's is almost always a contraction ("is"), not a possessive
contraction_s_words = {"it", "he", "she", "what", "that", "there",
                       "who", "here", "how", "when", "where"}

contractions = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have", "'m": " am",
}


def clean_text(text):
    text = text.lower()

    # Handle 's separately, based on the word right before it
    def expand_apostrophe_s(match):
        word = match.group(1)
        if word in contraction_s_words:
            return word + " is"
        return word  # possessive: drop the 's, keep the base word

    text = re.sub(r"\b(\w+)'s\b", expand_apostrophe_s, text)

    # Apply the rest of the contraction rules
    for pattern, replacement in contractions.items():
        text = text.replace(pattern, replacement)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)
    return text
