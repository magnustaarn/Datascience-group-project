import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Keep rows with required target/text columns
required_cols = ["stemmed_text", "label"]
df_train = df_train.dropna(subset=required_cols).copy()
df_test = df_test.dropna(subset=required_cols).copy()

# If column does not exist, fill with empty string
text_meta_cols = ["title", "summary", "meta_description", "authors"]
for col in text_meta_cols:
    if col not in df_train.columns:
        df_train[col] = ""
    if col not in df_test.columns:
        df_test[col] = ""
    df_train[col] = df_train[col].fillna("").astype(str)
    df_test[col] = df_test[col].fillna("").astype(str)

# If date does not exist, fill with pd.NaT (missing date value)
date_cols = ["scraped_at", "inserted_at", "updated_at"]
for col in date_cols:
    if col not in df_train.columns:
        df_train[col] = pd.NaT
    if col not in df_test.columns:
        df_test[col] = pd.NaT
    df_train[col] = pd.to_datetime(df_train[col], errors="coerce")
    df_test[col] = pd.to_datetime(df_test[col], errors="coerce")

# Converts date to year and month
for df in [df_train, df_test]:
    for col in date_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month

# Columns in model
feature_cols = [
    "stemmed_text",
    "title",
    "summary",
    "meta_description",
    "authors",
    "scraped_at_year",
    "scraped_at_month",
    "inserted_at_year",
    "inserted_at_month",
    "updated_at_year",
    "updated_at_month",
]

X_train = df_train[feature_cols]
y_train = df_train["label"]

X_test = df_test[feature_cols]
y_test = df_test["label"]

# Text columns which use bag of words conversion
text_features = []

candidate_text_cols = [
    ("stemmed_text_bow", "stemmed_text", 10000),
    ("title_bow", "title", 3000),
    ("summary_bow", "summary", 3000),
    ("meta_desc_bow", "meta_description", 3000),
    ("authors_bow", "authors", 1000),
]

for name, col, max_feat in candidate_text_cols:
    if col in df_train.columns:
        s = df_train[col].fillna("").astype(str)

        # only include column if it has usable words
        if s.str.contains(r"[A-Za-z]{2,}", regex=True).sum() > 0:
            text_features.append((name, CountVectorizer(max_features=max_feat), col))
            print(f"Included text feature: {col}")
        else:
            print(f"Skipped text feature: {col} (no usable vocabulary)")

# Numeric metadata
numeric_features = [
    "scraped_at_year",
    "scraped_at_month",
    "inserted_at_year",
    "inserted_at_month",
    "updated_at_year",
    "updated_at_month",
]

# Numeric transformer
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing values
    ("scaler", StandardScaler())  # standardize features
])

# Combine text and numeric preprocessing
preprocessor = ColumnTransformer(
    transformers=text_features + [
        ("num", numeric_transformer, numeric_features)
    ],
    remainder="drop"
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("F1 score:", f1_score(y_test, y_pred, average="binary"))
print(classification_report(y_test, y_pred))