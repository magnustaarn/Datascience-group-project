import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter
import matplotlib.pyplot as plt

def clean_text(text):
    text = str(text)
    cleaned_text = text.lower() # highercase -> lowercase
    cleaned_text = re.sub(r"\s+", " ", cleaned_text) # line shift, several spaces and tab
    cleaned_text = re.sub(r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<DATE>", cleaned_text) # date with time
    cleaned_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "DATE_TOKEN", cleaned_text) # date
    cleaned_text = re.sub(r"\d[\d,\.\S*]*", "NUM_TOKEN", cleaned_text) # any number
    cleaned_text = re.sub(r"\S+@\S+", "EMAIL_TOKEN", cleaned_text)  # mail
    cleaned_text = re.sub(r"(https?://\S+|www\.\S+)", "URL_TOKEN", cleaned_text)  # URL
    cleaned_text = re.sub(r"[.:,;\?=\+\-\*’\"\–\“\|\)\(\"\!\”\‘\$\—]", "", cleaned_text)  # removes punctation
    cleaned_text = re.sub(r"'s", "", cleaned_text)  # removes 's
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# tokenize text
def tokenize(text):
    return word_tokenize(text)


# removes stopwords
dff = pd.read_csv("englishST.txt", header=None, names=["stopword"])
stopwords = set(dff["stopword"])
def remove_stopwords(token, stopwords):
    return [w for w in token if w not in stopwords]

# stemming
stemmer = SnowballStemmer("english")
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# makes dictionary(with frequencies)
def build_dictionary(token_lists):
    return Counter(token for row in token_lists for token in row)

# plot of n most frequent words. Bar chart, only usable up to about 100 words
def plot_most_frequent_words_from_dict(freqs, n_words=100):
    top_words = freqs.most_common(n_words)

    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]

    plt.bar(words, counts)
    plt.xticks(rotation=90)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(f"Top {n_words} Most Frequent Words")

# plot word rank and frequency. Usable for showing many word frequencies
def plot_most_frequent_words(freqs, n_words=10000):
    top_words = freqs.most_common(n_words)

    counts = [count for _, count in top_words] # only keep frequency

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(counts) + 1), counts) #plot rank (rank1 = most frequent word) and frequency
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Top {n_words} Most Frequent Words")
    plt.tight_layout()
    plt.show()

# pipeline of whole data preprocessing. stores intermediates
def data_pipeline(df):
    df["cleaned_text"] = df["content"].apply(clean_text)
    df["tokenized_text"] = df["cleaned_text"].apply(tokenize)
    df["token_without_stopwords"] = df["tokenized_text"].apply(
    lambda tokens: remove_stopwords(tokens, stopwords)
    )
    df["stemmed_text"] = df["token_without_stopwords"].apply(stemming)
    return df

# pipeline of whole data preprocessing. only stores fully processed data
# should be faster when the tokenized but non stemmed data is not needed
def data_pipeline_stemmed(df):
    result = []

    for text in df["content"].fillna(""):
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        tokens = remove_stopwords(tokens, stopwords)
        stems = stemming(tokens)
        result.append(stems)

    return result

# data processing of news_sample.csv
def main():
    df = pd.read_csv("news_sample.csv")

    df = data_pipeline(df)

    dictionary_tokenized = build_dictionary(df["token_without_stopwords"])
    dictionary_stemmed = build_dictionary(df["stemmed_text"])
    dictionary_non_tokenized = (word for row in df["cleaned_text"] for word in row.split())
    plot_most_frequent_words(dictionary_tokenized)

    print("Dictionary size of tokenized and stopwords removed:", len(dictionary_tokenized))
    print("dictionary size of nontokenized and no stopword removal:", len(dictionary_non_tokenized))
    print("Size reduction:", len(dictionary_non_tokenized) - len(dictionary_tokenized))
    print("Dictionary size after stemming:", len(dictionary_stemmed))
    print("Reduction rate after stemming:",
        100 - (len(dictionary_stemmed) / len(dictionary_tokenized) * 100))
    

# only runs main() if file ran directly (this file usable as module)
if __name__ == "__main__":
    main()

