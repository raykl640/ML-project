"""
Crop Recommendation System — Team Presentation Dashboard
One section per teammate's part of the pipeline.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
)

# Palette — muted earth tones, one hue per pipeline stage
COLORS = {
    "forest":  "#3A4A38",
    "slate":   "#47607C",
    "plum":    "#6B4C63",
    "ochre":   "#96702F",
    "sage":    "#4B6350",
    "teal":    "#3F6B63",
    "rust":    "#8B4A3B",
    "deepteal": "#2E5266",
}

BANNER_COLORS = {
    "overview":  COLORS["forest"],
    "cleaning":  COLORS["slate"],
    "eda":       COLORS["plum"],
    "features":  COLORS["ochre"],
    "builder1":  COLORS["sage"],
    "builder2":  COLORS["teal"],
    "eval":      COLORS["rust"],
    "demo":      COLORS["deepteal"],
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #F4F5EF;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    color: #232B21;
}

.section-banner {
    padding: 1.3rem 1.6rem;
    border-radius: 6px;
    margin-bottom: 1.6rem;
    color: #F6F7F1;
    border-left: 4px solid rgba(255,255,255,0.35);
}
.section-banner .role-tag {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.75;
}
.section-banner h2 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    margin: 0.3rem 0 0 0;
    color: #FBFCF7;
    font-size: 1.55rem;
}

.insight-box {
    background: #FFFFFF;
    border: 1px solid #DDE1D6;
    border-left: 3px solid #4B6350;
    border-radius: 5px;
    padding: 1.1rem 1.4rem;
    margin-top: 0.5rem;
}
.insight-box p {
    margin: 0 0 0.5rem 0;
    color: #3A4235;
    line-height: 1.6;
    font-size: 0.94rem;
}
.insight-box p:last-child { margin-bottom: 0; }
.insight-label {
    display: block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B6350;
    margin-bottom: 0.55rem;
}

div[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif;
    font-size: 1.65rem;
    color: #232B21;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    color: #5B6656;
}

section[data-testid="stSidebar"] {
    background-color: #ECEEE4;
    border-right: 1px solid #DBDFD3;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.25rem;
}
</style>
""", unsafe_allow_html=True)


def banner(key, role, title):
    st.markdown(
        f"""<div class="section-banner" style="background:{BANNER_COLORS[key]}">
        <span class="role-tag">{role}</span><h2>{title}</h2></div>""",
        unsafe_allow_html=True,
    )


def insight(label, *paragraphs):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(
        f"""<div class="insight-box"><span class="insight-label">{label}</span>{body}</div>""",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    return pd.read_csv("data/crop_recommendation_cleaned.csv")


@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    rf = joblib.load("models/rf_model.pkl")
    gb = joblib.load("models/gb_model.pkl")
    svm = joblib.load("models/svm_model.pkl")
    knn = joblib.load("models/knn_model.pkl")
    voting = joblib.load("models/voting_ensemble.pkl")
    stacking = joblib.load("models/stacking_ensemble.pkl")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test.pkl")
    return {
        "scaler": scaler, "encoder": encoder,
        "models": {
            "Random Forest": rf, "Gradient Boosting": gb,
            "SVM": svm, "KNN": knn,
            "Voting Ensemble": voting, "Stacking Ensemble": stacking,
        },
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


df = load_data()
bundle = load_models()
encoder = bundle["encoder"]
scaler = bundle["scaler"]
models = bundle["models"]
X_test, y_test = bundle["X_test"], bundle["y_test"]

NUMERIC_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@st.cache_data
def compute_metrics():
    rows = []
    for name, m in models.items():
        pred = m.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (macro)": precision_score(y_test, pred, average="macro"),
            "Recall (macro)": recall_score(y_test, pred, average="macro"),
            "F1 (macro)": f1_score(y_test, pred, average="macro"),
        })
    return pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)


metrics_df = compute_metrics()

# Sidebar navigation
st.sidebar.title("Crop Recommendation")
st.sidebar.caption("ML Project — Team Presentation")
st.sidebar.markdown(
    f'<div style="height:5px;border-radius:3px;margin:0.3rem 0 1.1rem 0;'
    f'background:linear-gradient(90deg,{COLORS["forest"]},{COLORS["slate"]},'
    f'{COLORS["plum"]},{COLORS["ochre"]},{COLORS["sage"]},{COLORS["teal"]},'
    f'{COLORS["rust"]},{COLORS["deepteal"]});"></div>',
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "Project section",
    [
        "Overview",
        "1 · Data Cleaning",
        "2 · EDA & Visualization",
        "3 · Feature Engineering",
        "4 · Model Builder #1 (RF & GB)",
        "5 · Model Builder #2 (SVM/KNN + Ensembles)",
        "6 · Evaluation",
        "7 · Live Demo",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Each section reflects one teammate's part of the pipeline, "
    "from raw data to a deployed prediction demo."
)

# Overview
if section == "Overview":
    banner("overview", "Project", "Machine Learning-Based Crop Recommendation System")
    st.write(
        "Recommends the most suitable crop for a plot of land based on soil "
        "nutrients and environmental conditions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{df.shape[0]:,}")
    c2.metric("Crop classes", df["label"].nunique())
    c3.metric("Input features", len(NUMERIC_COLS))
    c4.metric("Best model F1 (macro)", f"{metrics_df.iloc[0]['F1 (macro)']:.3f}")

    st.markdown("#### How the work was divided")
    st.dataframe(
        pd.DataFrame({
            "Section": [
                "Data Cleaning", "EDA & Visualization", "Feature Engineering",
                "Model Builder #1", "Model Builder #2", "Evaluation", "Deployment",
            ],
            "Focus": [
                "Load, validate, and clean the raw dataset",
                "Explore distributions and relationships in the data",
                "Engineer features, scale, and split train/test",
                "Tune Random Forest & Gradient Boosting",
                "Tune SVM & KNN, assemble Voting/Stacking ensembles",
                "Compare all models on held-out test data",
                "Ship an interactive prediction demo",
            ],
        }),
        hide_index=True, width='stretch',
    )

# Data cleaning
elif section == "1 · Data Cleaning":
    banner("cleaning", "Data Lead", "Data Cleaning")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing values per column**")
        st.dataframe(df.isnull().sum().rename("nulls"), width='stretch')
    with col2:
        st.markdown("**Negative / invalid value check**")
        st.dataframe((df[NUMERIC_COLS] < 0).sum().rename("negative count"), width='stretch')

    st.markdown(f"**Duplicate rows:** {df.duplicated().sum()}")

    st.markdown("**Sample of cleaned data**")
    st.dataframe(df.head(10), width='stretch')

    insight(
        "Summary",
        "The dataset loaded cleanly, with no missing values and no duplicate "
        "records across any of the rows.",
        "Data types matched the expected schema, and all numeric fields fell "
        "within valid, non-negative ranges. The cleaned file was exported as "
        "<code>crop_cleaned.csv</code>.",
    )

# EDA
elif section == "2 · EDA & Visualization":
    banner("eda", "EDA / Visualization Lead", "Exploratory Data Analysis")

    chart = st.selectbox(
        "Choose a chart",
        [
            "Crop distribution",
            "Numeric feature distributions",
            "Correlation heatmap",
            "Average conditions per crop",
            "Rainfall distribution by crop",
            "Temperature vs Humidity",
            "Rainfall vs Temperature",
            "Total NPK distribution",
            "Average NPK per crop",
        ],
    )

    if chart == "Crop distribution":
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.countplot(x="label", data=df, ax=ax, color=COLORS["sage"])
        ax.set_title("Distribution of Crop Types")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif chart == "Numeric feature distributions":
        df[NUMERIC_COLS].hist(figsize=(14, 8), bins=20, color=COLORS["sage"])
        plt.tight_layout()
        st.pyplot(plt.gcf())

    elif chart == "Correlation heatmap":
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(df[NUMERIC_COLS].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif chart == "Average conditions per crop":
        crop_mean = df.groupby("label")[NUMERIC_COLS].mean()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(crop_mean, cmap="YlGnBu", ax=ax)
        ax.set_title("Average Conditions Required by Each Crop")
        st.pyplot(fig)

    elif chart == "Rainfall distribution by crop":
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.boxplot(x="label", y="rainfall", data=df, ax=ax)
        plt.xticks(rotation=90)
        ax.set_title("Rainfall Distribution by Crop")
        st.pyplot(fig)

    elif chart == "Temperature vs Humidity":
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=df, x="temperature", y="humidity", hue="label", ax=ax, legend=False)
        ax.set_title("Temperature vs Humidity")
        st.pyplot(fig)

    elif chart == "Rainfall vs Temperature":
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=df, x="rainfall", y="temperature", hue="label", ax=ax, legend=False)
        ax.set_title("Rainfall vs Temperature")
        st.pyplot(fig)

    elif chart == "Total NPK distribution":
        npk = df["N"] + df["P"] + df["K"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(npk, bins=20, kde=True, ax=ax, color=COLORS["ochre"])
        ax.set_title("Distribution of Total NPK")
        st.pyplot(fig)

    elif chart == "Average NPK per crop":
        npk_crop = (df["N"] + df["P"] + df["K"]).groupby(df["label"]).mean().sort_values()
        fig, ax = plt.subplots(figsize=(14, 5))
        npk_crop.plot(kind="bar", ax=ax, color=COLORS["ochre"])
        ax.set_title("Average Total NPK by Crop")
        st.pyplot(fig)

    insight(
        "Key takeaways",
        "Crop classes are evenly represented, which lowers the risk of the "
        "model favoring any single label.",
        "Soil nutrients and environmental conditions vary meaningfully across "
        "crop types, and a handful of the environmental variables show "
        "moderate correlation with one another.",
    )

# Feature engineering
elif section == "3 · Feature Engineering":
    banner("features", "Feature Engineering Lead", "Feature Engineering & Data Preparation")

    st.markdown(
        "This stage prepares the cleaned dataset for modeling:\n"
        "- Separate features (`X`) and target (`label`)\n"
        "- Encode crop labels with `LabelEncoder`\n"
        "- Engineer two new features: `NPK_sum = N + P + K` and "
        "`NPK_ratio = N / (P + K + 1)`\n"
        "- Scale all 9 features with `StandardScaler`\n"
        "- Split into train/test sets (80/20, stratified by crop)"
    )

    X_train = bundle["X_train"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Training samples", X_train.shape[0])
    c2.metric("Test samples", X_test.shape[0])
    c3.metric("Final feature count", X_train.shape[1])

    st.markdown("**Final feature set (in model input order):**")
    st.code(", ".join(scaler.feature_names_in_), language="text")

    demo = df[NUMERIC_COLS].head(5).copy()
    demo["NPK_sum"] = demo["N"] + demo["P"] + demo["K"]
    demo["NPK_ratio"] = demo["N"] / (demo["P"] + demo["K"] + 1)
    st.markdown("**Example of engineered features:**")
    st.dataframe(demo, width='stretch')

    insight(
        "Outputs shared with the team",
        "<code>scaler.pkl</code>, <code>label_encoder.pkl</code>, and "
        "<code>train_test.pkl</code>.",
    )

# Model builder 1
elif section == "4 · Model Builder #1 (RF & GB)":
    banner("builder1", "Model Builder #1", "Random Forest & Gradient Boosting")

    st.markdown("Both models were tuned with `GridSearchCV` (5-fold stratified CV), scored on **macro-F1**.")

    for name in ["Random Forest", "Gradient Boosting"]:
        row = metrics_df[metrics_df["Model"] == name].iloc[0]
        st.markdown(f"**{name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
        c2.metric("Precision", f"{row['Precision (macro)']:.4f}")
        c3.metric("Recall", f"{row['Recall (macro)']:.4f}")
        c4.metric("F1 (macro)", f"{row['F1 (macro)']:.4f}")
        st.markdown("---")

    with st.expander("Tuned hyperparameters"):
        st.write("**Random Forest:**", models["Random Forest"].get_params())
        st.write("**Gradient Boosting:**", models["Gradient Boosting"].get_params())

# Model builder 2
elif section == "5 · Model Builder #2 (SVM/KNN + Ensembles)":
    banner("builder2", "Model Builder #2", "SVM / KNN + Ensemble Assembly")

    st.markdown(
        "SVM and KNN were tuned the same way as the RF/GB models above, then combined with "
        "all four base learners into a **soft Voting** ensemble and a **Stacking** ensemble "
        "(logistic regression meta-learner, fit on out-of-fold predictions)."
    )

    for name in ["SVM", "KNN", "Voting Ensemble", "Stacking Ensemble"]:
        row = metrics_df[metrics_df["Model"] == name].iloc[0]
        st.markdown(f"**{name}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{row['Accuracy']:.4f}")
        c2.metric("Precision", f"{row['Precision (macro)']:.4f}")
        c3.metric("Recall", f"{row['Recall (macro)']:.4f}")
        c4.metric("F1 (macro)", f"{row['F1 (macro)']:.4f}")
        st.markdown("---")

    st.markdown("#### Voting vs Stacking vs individual base learners")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = metrics_df.set_index("Model")["F1 (macro)"].sort_values()
    plot_df.plot(kind="barh", ax=ax, color=COLORS["teal"])
    ax.set_xlabel("F1 (macro)")
    st.pyplot(fig)

# Evaluation
elif section == "6 · Evaluation":
    banner("eval", "Evaluation Lead", "Model Evaluation")

    st.markdown("**Full comparison — all models on the held-out test set:**")
    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.4f}", "Precision (macro)": "{:.4f}",
            "Recall (macro)": "{:.4f}", "F1 (macro)": "{:.4f}",
        }).background_gradient(subset=["F1 (macro)"], cmap="Greens"),
        hide_index=True, width='stretch',
    )

    st.markdown("#### Confusion matrix")
    model_choice = st.selectbox("Model", list(models.keys()), index=list(models.keys()).index("Stacking Ensemble"))
    pred = models[model_choice].predict(X_test)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_choice} — Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    st.pyplot(fig)

    with st.expander("Full classification report"):
        st.text(classification_report(y_test, pred, target_names=encoder.classes_))

# Live demo
elif section == "7 · Live Demo":
    banner("demo", "Deployment Lead", "Try the Crop Recommender")

    st.markdown("Enter soil and environmental conditions to get a live crop recommendation.")

    model_choice = st.selectbox(
        "Model to use for prediction",
        list(models.keys()),
        index=list(models.keys()).index("Stacking Ensemble"),
    )

    d = df[NUMERIC_COLS].describe()

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Nitrogen (N)", float(d["N"]["min"]), float(d["N"]["max"]), float(d["N"]["mean"]))
        p = st.slider("Phosphorus (P)", float(d["P"]["min"]), float(d["P"]["max"]), float(d["P"]["mean"]))
    with c2:
        k = st.slider("Potassium (K)", float(d["K"]["min"]), float(d["K"]["max"]), float(d["K"]["mean"]))
        temp = st.slider("Temperature (°C)", float(d["temperature"]["min"]), float(d["temperature"]["max"]), float(d["temperature"]["mean"]))
    with c3:
        hum = st.slider("Humidity (%)", float(d["humidity"]["min"]), float(d["humidity"]["max"]), float(d["humidity"]["mean"]))
        ph = st.slider("Soil pH", float(d["ph"]["min"]), float(d["ph"]["max"]), float(d["ph"]["mean"]))
    rain = st.slider("Rainfall (mm)", float(d["rainfall"]["min"]), float(d["rainfall"]["max"]), float(d["rainfall"]["mean"]))

    if st.button("Get recommendation", type="primary"):
        npk_sum = n + p + k
        npk_ratio = n / (p + k + 1)
        row = pd.DataFrame([{
            "N": n, "P": p, "K": k, "temperature": temp, "humidity": hum,
            "ph": ph, "rainfall": rain, "NPK_sum": npk_sum, "NPK_ratio": npk_ratio,
        }])[list(scaler.feature_names_in_)]

        scaled = scaler.transform(row)
        model = models[model_choice]
        pred_idx = model.predict(scaled)[0]
        crop = encoder.inverse_transform([pred_idx])[0]

        st.markdown(
            f"""<div class="insight-box" style="border-left-color:{COLORS['deepteal']}">
            <span class="insight-label">Recommended crop</span>
            <p style="font-family:'Fraunces',serif;font-size:1.4rem;color:#232B21;">
            {crop.capitalize()}</p></div>""",
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            top5 = pd.Series(proba, index=encoder.classes_).sort_values(ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(7, 3))
            top5.iloc[::-1].plot(kind="barh", ax=ax, color=COLORS["deepteal"])
            ax.set_xlabel("Predicted probability")
            ax.set_title("Top 5 candidate crops")
            st.pyplot(fig)
