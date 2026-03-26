import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
from paths import DATA_DIR

train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"
val_file = DATA_DIR / "validation.csv"
#Load train/val/test
print("Loading data")
df_train = pd.read_csv(train_file, usecols=["stemmed_text", "label"])
df_val = pd.read_csv(val_file, usecols=["stemmed_text", "label"])
df_test = pd.read_csv(test_file, usecols=["stemmed_text", "label"])


#Remove NaN
print("Removing NaN")
df_train = df_train.dropna(subset=["stemmed_text", "label"])
df_val = df_val.dropna(subset=["stemmed_text", "label"])
df_test = df_test.dropna(subset=["stemmed_text", "label"])

#TF-IDF Vectorizer
print("Starting TF-IDF")
vectorizer = TfidfVectorizer(
  max_features = 20000, #Eller 10000?
  ngram_range = (1,2),
  min_df = 5,
  max_df = 0.8,
)

#Fit and transform
print("Starting vectorizer...")
X_train = vectorizer.fit_transform(df_train["stemmed_text"])
X_val = vectorizer.transform(df_val["stemmed_text"])
X_test = vectorizer.transform(df_test["stemmed_text"])
print("Vectorizer finished")      

y_train = df_train["label"]
y_val = df_val["label"]
y_test = df_test["label"]

#SVM Model
print("Training model...")
model = LinearSVC(class_weight="balanced", max_iter=10000)
model.fit(X_train, y_train)
print("Model finished")

#Evaluation
print("\nValidation:")
y_val_pred = model.predict(X_val)
print("F1 score:", f1_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

print("\nTest:")
y_test_pred = model.predict(X_test)
print("F1 score:", f1_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

                  







