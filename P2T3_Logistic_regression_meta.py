import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler


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
text_features = [
    ("stemmed_text_bow", CountVectorizer(max_features=10000), "stemmed_text"), # max 10.000 dict size
    ("title_bow", CountVectorizer(max_features=3000), "title"), # max 3.000 dict size
    ("summary_bow", CountVectorizer(max_features=3000), "summary"), # max 3.000 dict size
    ("meta_desc_bow", CountVectorizer(max_features=3000), "meta_description"), # max 3.000 dict size
    ("authors_bow", CountVectorizer(max_features=1000), "authors"), # max 1.000 dict size
]

# Numeric metadata 
numeric_features = [
    "scraped_at_year",
    "scraped_at_month",
    "inserted_at_year",
    "inserted_at_month",
    "updated_at_year",
    "updated_at_month",
]

# numeric transformer
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # fill missing values
    ("scaler", StandardScaler()) # standerdises features (None of the features should be significantly higher)
])

# 
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


print("F1 score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))