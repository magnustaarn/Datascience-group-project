from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd
import json
import joblib

# load train.csv & validation.csv
print("Loading data...")
df_train = pd.read_csv("train.csv", usecols=['stemmed_text', 'label'])
df_val = pd.read_csv("validation.csv", usecols=['stemmed_text', 'label'])
df_train = df_train.dropna(subset=['stemmed_text'])
df_val = df_val.dropna(subset=['stemmed_text'])

# TF-IDF weights - 20.000 best words
print("Vectorizing text (TF-IDF)...")
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5, # ignores words that appear in less than 5 articles
    max_df=0.8 # ignores words that appear in 80%+ of articles
)
X_train = tfidf.fit_transform(df_train['stemmed_text'])
X_val = tfidf.transform(df_val['stemmed_text'])

# adjusting the model
h_layers = (64, 32)
m_iter = 50

# neural network model
print("Training MLP (Neural Network)...")
mlp = MLPClassifier(
    hidden_layer_sizes=h_layers,
    max_iter=m_iter,
    verbose=True, # live update on loss through each iteration
    early_stopping=True, # if the model has learned enough, it stops with further iterations
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
mlp.fit(X_train, df_train['label'])

# Name of json file
model_name = f"loss_{h_layers}".replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")

# Save loss_curve_ as JSON file for graph
with open(f"{model_name}.json", "w") as f:
    json.dump(mlp.loss_curve_, f)

print(f"Model saved as {model_name}.json")

# Evaluation
print("Evaluating on validation set...")
y_pred = mlp.predict(X_val)
print(classification_report(df_val['label'], y_pred))

joblib.dump(mlp, "trained_mlp_model.pkl") # save final model
joblib.dump(tfidf, "tfidf_vectorizer.pkl") # save TF-IDF vectorizer

print("Model & Vectorizer saved as .pkl files")