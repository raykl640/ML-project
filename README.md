# 🌱 Crop Recommendation System — Presentation Dashboard

A Streamlit app with one section per teammate's part of the project:
Data Cleaning → EDA → Feature Engineering → Model Builder #1 (RF/GB) →
Model Builder #2 (SVM/KNN + Ensembles) → Evaluation → Live Demo.

## Run it locally (optional, to check before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Deploy for free — shareable link (Streamlit Community Cloud)

**1. Push this folder to GitHub**

- Go to https://github.com/new and create a new repository (public or private),
  e.g. `crop-recommendation-app`.
- Upload all the files in this folder (`app.py`, `requirements.txt`, `data/`, `models/`)
  either by dragging them into the GitHub web UI ("Add file → Upload files"),
  or via git:

```bash
cd crop-app
git init
git add .
git commit -m "Crop recommendation presentation app"
git branch -M main
git remote add origin https://github.com/<your-username>/crop-recommendation-app.git
git push -u origin main
```

**2. Deploy on Streamlit Community Cloud**

- Go to https://share.streamlit.io and sign in with your GitHub account.
- Click **"New app"**.
- Select your repo, branch `main`, and main file path `app.py`.
- Click **Deploy**.

That's it — Streamlit installs `requirements.txt` automatically and gives you a
public URL like `https://your-app-name.streamlit.app` you can share with anyone
for the presentation.

First deploy takes 2–3 minutes (installing scikit-learn/matplotlib). After that,
any future `git push` to the repo auto-redeploys the app.

## Folder structure

```
crop-app/
├── app.py                  # the Streamlit app (all 8 sections)
├── requirements.txt
├── data/
│   └── crop_recommendation_cleaned.csv
└── models/
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── train_test.pkl
    ├── rf_model.pkl
    ├── gb_model.pkl
    ├── svm_model.pkl
    ├── knn_model.pkl
    ├── voting_ensemble.pkl
    └── stacking_ensemble.pkl
```

## Notes

- `requirements.txt` pins `scikit-learn==1.6.1` to match the version the models
  were trained with — don't remove that pin, or you may hit a version-mismatch
  error unpickling the models.
- The `models/` folder is ~60MB total, which is well within GitHub's per-file
  and repo size limits — no Git LFS needed.
- If you retrain any model later, just drop the new `.pkl` into `models/`
  (same filename) and push — no code changes needed.
