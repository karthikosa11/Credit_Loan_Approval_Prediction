import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Loan Approval", layout="centered")
st.title("Credit Loan Approval Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head(15), use_container_width=True)

# ✅ Change this if your target column name is different
TARGET = "Loan_Status"

if TARGET not in df.columns:
    st.error(f"TARGET column '{TARGET}' not found. Available columns: {list(df.columns)}")
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Column types
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

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

st.subheader("Model (Baseline)")
st.write(f"Accuracy: **{acc:.3f}**")

st.subheader("Try a Prediction")

user_input = {}
for col in X.columns:
    if col in num_cols:
        default = float(df[col].dropna().median()) if df[col].notna().any() else 0.0
        user_input[col] = st.number_input(col, value=default)
    else:
        options = sorted(df[col].dropna().unique().tolist())
        user_input[col] = st.selectbox(col, options if options else [""])

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    out = clf.predict(input_df)[0]
    st.success(f"Prediction: **{out}**")
