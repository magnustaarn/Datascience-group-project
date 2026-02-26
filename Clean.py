import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

df = pd.read_csv("news_sample.csv") # read CSV
dff = pd.read_csv("englishST.txt", header=None, names=["stopword"])

def clean_text(text):
    cleaned_text = text.lower() # highercase -> lowercase
    cleaned_text = re.sub(r"\s+", " ", cleaned_text) # line shift, several spaces and tab
    cleaned_text = re.sub(r"\d[\d,\.\S*]*", "NUM_TOKEN", cleaned_text) # any number
    cleaned_text = re.sub(r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<DATE>", cleaned_text) # date with time
    cleaned_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "DATE_TOKEN", cleaned_text) # date
    cleaned_text = re.sub(r"\S+@\S+", "EMAIL_TOKEN", cleaned_text)  # mail
    cleaned_text = re.sub(r"(https?://\S+|www\.\S+)", "URL_TOKEN", cleaned_text)  # URL
    cleaned_text = re.sub(r"[.:,;\?=\+\-\*’\"\–\“\|]", "", cleaned_text)  # removes punctation
    cleaned_text = re.sub(r"'s", "", cleaned_text)  # removes 's
    cleaned_text = cleaned_text.strip()
    return cleaned_text

df["cleaned_text"] = df["content"].apply(clean_text) # clean specific column
df["tokenized_text"] = df["cleaned_text"].apply(word_tokenize) # tokenized text
stopwords = set(dff["stopword"]) # makes a set of stopwords

# removes stopwords from tokenized text
df["token_without_stopwords"] = df["tokenized_text"].apply(
    lambda tokens: [w for w in tokens if w not in stopwords]
)

stemmer = SnowballStemmer("english") #choose language for stemming
# stem each word in tokenized 
df["stemmed_text"] = df["token_without_stopwords"].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]
)

# creates dictionary(set) of tokenized and stopword removed text
dictionary_tokenized_no_stopwords = set(
    token for row in df["token_without_stopwords"] for token in row
)

# creates dictionary(set) of non tokenized and no stopword removal
dictionary_non_tokenized_without_stopwords = set(
    word for row in df["cleaned_text"] for word in row.split()
)


# creates dictionary(set) of stemmed, tokenized, and stopword removed text
dictionary_stemmed = set(
    token for row in df["stemmed_text"] for token in row
)


print("dictionary size of tokenized and stopwords removed", len(dictionary_tokenized_no_stopwords)) # print dictionary size of tokenized and stopword removal
print("dictionary size of nontokenized and no stopword removal:", len(dictionary_non_tokenized_without_stopwords)) # print dictionary size of non-tokenized and no stopword removal
print("size reduction after tokenization and stopword removal:",len(dictionary_non_tokenized_without_stopwords)-len(dictionary_tokenized_no_stopwords)) # print size reduction after tokenization and stopword removal
print("dictionary size after stemming:", len(dictionary_stemmed))
print("Reduction rate after stemming:", 100-(len(dictionary_stemmed)/len(dictionary_tokenized_no_stopwords)*100)) # prints the reduction rate after stemming