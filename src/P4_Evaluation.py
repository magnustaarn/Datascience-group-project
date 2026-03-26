import pandas as pd
import joblib
import P1T1_Cleaning as clean # for cleaning LIAR - so the evaluation is fair against both datasets
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from paths import DATA_DIR, OUTPUT_DIR_FIGURES

# file paths
mlp_file = DATA_DIR / "trained_mlp_model.pkl"
tfidf_file = DATA_DIR / "tfidf_vectorizer.pkl"
log_model_file = DATA_DIR / "simple_log_model.pkl"
count_vec_file = DATA_DIR / "simple_count_vectorizer.pkl"

test_file = DATA_DIR / "test.csv"
liar_test_file = DATA_DIR / "LIAR_test.tsv"

cm_fn_mlp_file = OUTPUT_DIR_FIGURES / "cm_fn_mlp.png"
cm_fn_log_file = OUTPUT_DIR_FIGURES / "cm_fn_log.png"
cm_liar_mlp_file = OUTPUT_DIR_FIGURES / "cm_liar_mlp.png"
cm_liar_log_file = OUTPUT_DIR_FIGURES / "cm_liar_log.png"

# loading models and vectorizations
print("Loading models...")
mlp = joblib.load(mlp_file)
tfidf = joblib.load(tfidf_file)
log_model = joblib.load(log_model_file)
count_vec = joblib.load(count_vec_file)

# FakeNewsCorpus dataset
print("\nFakeNewsCorpus Evaluation")
df_fn = pd.read_csv(test_file).dropna(subset=['stemmed_text', 'label'])

X_fn_tfidf = tfidf.transform(df_fn['stemmed_text'])
y_fn_pred_mlp = mlp.predict(X_fn_tfidf)

X_fn_count = count_vec.transform(df_fn['stemmed_text'])
y_fn_pred_log = log_model.predict(X_fn_count)

print("FakeNewsCorpus - MLP result:\n", classification_report(df_fn['label'], y_fn_pred_mlp))
print("FakeNewsCorpus - Logistic result:\n", classification_report(df_fn['label'], y_fn_pred_log))

# LIAR dataset
print("\nLIAR Dataset Evaluation")
liar_cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
df_liar = pd.read_csv(liar_test_file, sep='\t', names=liar_cols, header=None)

# MAP labels (0=Fake, 1=Reliable)
label_map = {
    'pants-fire': 0, 'false': 0, 'barely-true': 0,
    'half-true': 1, 'mostly-true': 1, 'true': 1
}
df_liar['binary_label'] = df_liar['label'].map(label_map)
df_liar = df_liar.dropna(subset=['statement', 'binary_label'])

# processing LIAR dataset (cleaning, stemming)
print("Cleaning and stemming LIAR statements...")
df_liar_temp = df_liar.rename(columns={'statement': 'content'})
liar_tokens_list = clean.data_pipeline_stemmed(df_liar_temp)
df_liar['stemmed_text'] = [" ".join(tokens) for tokens in liar_tokens_list]

X_liar_tfidf = tfidf.transform(df_liar['stemmed_text'])
X_liar_count = count_vec.transform(df_liar['stemmed_text'])

y_liar_pred_mlp = mlp.predict(X_liar_tfidf)
y_liar_pred_log = log_model.predict(X_liar_count)

print("LIAR - MLP result:\n", classification_report(df_liar['binary_label'], y_liar_pred_mlp))
print("LIAR - Logistic result:\n", classification_report(df_liar['binary_label'], y_liar_pred_log))

# Confusion matrices
def plot_cm(y_true, y_pred, title, filename):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Fake', 'Reliable'], cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(filename)
    plt.show()

plot_cm(df_fn['label'], y_fn_pred_mlp, "FakeNews - MLP", cm_fn_mlp_file)
plot_cm(df_fn['label'], y_fn_pred_log, "FakeNews - Logistic", cm_fn_log_file)
plot_cm(df_liar['binary_label'], y_liar_pred_mlp, "LIAR - MLP", cm_liar_mlp_file)
plot_cm(df_liar['binary_label'], y_liar_pred_log, "LIAR - Logistic", cm_liar_log_file)
print("\nsaved all confusion matrix models in directory")

# Comparison
print("\nComparison")
print(f"FakeNews MLP: {f1_score(df_fn['label'], y_fn_pred_mlp):.2f}")
print(f"FakeNews Log: {f1_score(df_fn['label'], y_fn_pred_log):.2f}")
print(f"LIAR MLP: {f1_score(df_liar['binary_label'], y_liar_pred_mlp):.2f}")
print(f"LIAR Log: {f1_score(df_liar['binary_label'], y_liar_pred_log):.2f}")
