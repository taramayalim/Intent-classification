# Banking Intent Classifier (Money Movement)

Classical NLP system (TF-IDF + linear SVM) that classifies customer messages about
transfers and top-ups into 21 intents (BANKING77 subset).

## Files
- `app.py` - Streamlit dashboard
- `preprocess.py` - `clean_text()` shared with the training notebook
- `final_model.joblib`, `final_vectorizer.joblib`, `label_encoder.joblib` - saved by the notebook
- `requirements.txt`

## Run locally
1. Copy the three `.joblib` files (saved by the notebook's last cell) into this folder.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Deploy (Streamlit Community Cloud)
Push this folder, including the `.joblib` files, to a GitHub repo, then create a new app
on share.streamlit.io pointing at `app.py`.

## Important
Use the same scikit-learn version for training and deployment, otherwise the saved
files may not load. If you change `clean_text` or any tuning, retrain and re-save the
three model files, and update the metrics at the top of `app.py`.
