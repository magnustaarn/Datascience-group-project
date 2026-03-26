import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import joblib
from paths import DATA_DIR

train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"

df_train = pd.read_csv(train_file)
df_train = df_train.dropna(subset=["stemmed_text", "label"])# keep only rows with both text and label
X_train = df_train["stemmed_text"]
y_train = df_train["label"]

df_test = pd.read_csv(test_file)
df_test = df_test.dropna(subset=["stemmed_text", "label"])# keep only rows with both text and label
X_test = df_test["stemmed_text"]
y_test = df_test["label"]

# vectorize
vectorizer = CountVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("F1 score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model and vectorizer
model_file = OUTPUT_DIR / "simple_log_model.pkl"
vec_file = OUTPUT_DIR / "simple_count_vectorizer.pkl"
joblib.dump(model, model_file)
joblib.dump(vectorizer, vec_file)
print("Logistic model and vectorizer saved as .pkl files")