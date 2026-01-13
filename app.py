import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Credit Loan Approval", layout="centered")
st.title("Credit Loan Approval Prediction")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

# ---------------------------
# Target Column
# ---------------------------
TARGET = "Loan_Approved"

if TARGET not in df.columns:
    st.error(f"TARGET column '{TARGET}' not found. Available columns: {list(df.columns)}")
    st.stop()

# ---------------------------
# Clean Target (y) - FIX NaNs + normalize labels
# ---------------------------
# Keep rows with non-missing target
df = df[df[TARGET].notna()].copy()

# Normalize object labels like Yes/No to 1/0
if df[TARGET].dtype == "object":
    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
        })
    )

# Drop any rows that became NaN after mapping (unknown labels)
df = df[df[TARGET].notna()].copy()

# Force int 0/1
df[TARGET] = df[TARGET].astype(int)

# ---------------------------
# Split features/label
# ---------------------------
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Drop ID-like columns (not useful for prediction)
if "Applicant_ID" in X.columns:
    X = X.drop(columns=["Applicant_ID"])

# ---------------------------
# Train-Test Split
# ---------------------------
# Use stratify only if both classes exist
stratify_y = y if y.nunique() == 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_y
)

# ---------------------------
# Preprocessing
# ---------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# ---------------------------
# Model
# ---------------------------
model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# Train
clf.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

st.subheader("Model Performance (Baseline)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision", f"{prec:.3f}")
c3.metric("Recall", f"{rec:.3f}")
c4.metric("F1", f"{f1:.3f}")

# ---------------------------
# Prediction UI
# ---------------------------
st.subheader("Try a Prediction")

user_input = {}

for col in X.columns:
    if col in num_cols:
        # Use median as default
        default = float(df[col].dropna().median()) if df[col].notna().any() else 0.0
        user_input[col] = st.number_input(col, value=default)
    else:
        options = sorted(df[col].dropna().unique().tolist())
        user_input[col] = st.selectbox(col, options if options else [""])

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    out = int(clf.predict(input_df)[0])
    if out == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
