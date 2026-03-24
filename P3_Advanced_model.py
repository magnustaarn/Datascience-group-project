import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

#Load stopwords
stopword_df = pd.read_csv("englishST.txt", header=None, names["stopword"])
custom_stopwords = set(stopwords_df["stopword"])


#Load train/val/test
df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("train.csv")

#Remove NaN
df_train = df_train.dropna(subset=["content","label"])
df_val = df_val.dropna(subset=["content","label"])
df_test = df_test.dropna(subset=["content","label"])

#TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
  max_features = 20000,
  ngram_range = (1,2),
  min_df = 5,
  max_df = 0.8,
  stop_words = cusotm_stopwords
)

#Fit and transform
X_train = vectorizer.fit_transform(df_train["content"])
X_val = vectorizer.transform(df_val["content"])
X_test = vectorizer.transform(df_test["content"])       

y_train = df_train["label"]
y_val = df_val["label"]
y_test = df_test["label"]

#SVM Model
model = LinearSVC(class_weight="balanced", max_iter=10000)
model.fit(X_train, y_train)

#Evaluation
y_val_pred = model.pred(X_val)
print("Test F1 score:", f1_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = model.pred(X_test)
print("Test F1 score:", f1_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

                  







