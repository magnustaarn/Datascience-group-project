import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import Clean as clean

df = pd.read_csv("995,000_rows.csv")

df = clean.data_pipeline(df)

dictionary_tokenized = clean.build_dictionary(df["token_without_stopwords"])
dictionary_stemmed = clean.build_dictionary(df["stemmed_text"])
dictionary_non_tokenized = counter(
  word for row in df["cleaned_text"] for word in row.split()
)

#Counting
url_count = dictionary_non_tokenized["URL_TOKEN"]
date_count = dictionary_non_tokenized["DATE_TOKEN"]
num_count = dictionary_non_tokenized["NUM_TOKEN"]

#Frequency of words
dictionary_tokenized.most_common(100)

top100words = dictionary_tokenized.most_common(100)
for word, count in top100words:
  print(word, count)

clean.plot_most_frequent_words_from_dict(dictionary_tokenized, 100)
