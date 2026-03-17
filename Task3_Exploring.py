import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import Clean as clean

dict_tokenized = Counter()
dict_stemmed = Counter()
dict_non_tokenized = Counter()

# processing in chunk - faster run time
for chunk in pd.read_csv("995,000_rows.csv", chunksize=100000):
    
    chunk = clean.data_pipeline(chunk)
    
    # update with data from chunk
    dict_tokenized.update(clean.build_dictionary(chunk["token_without_stopwords"]))
    dict_stemmed.update(clean.build_dictionary(chunk["stemmed_text"]))

    for row in chunk["cleaned_text"]:
        dict_non_tokenized.update(row.split())

# Counting
url_count = dict_non_tokenized["URL_TOKEN"]
date_count = dict_non_tokenized["DATE_TOKEN"]
num_count = dict_non_tokenized["NUM_TOKEN"]

# Frequency of words
top100words = dict_tokenized.most_common(100)
for word, count in top100words:
    print(word, count)

# Plot
clean.plot_most_frequent_words_from_dict(dict_tokenized, 100)
clean.plot_most_frequent_words(dict_tokenized)

# Print directionaries
print("URL count:", url_count)
print("Date count:", date_count)
print("Number count:", num_count)

# Before and after
clean.plot_most_frequent_words_from_dict(dict_non_tokenized, 100)
clean.plot_most_frequent_words_from_dict(dict_stemmed, 100)
clean.plot_most_frequent_words_from_dict(dict_tokenized, 100)