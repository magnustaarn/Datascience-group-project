import pandas as pd
import Clean as clean
from collections import Counter
import os

dictionary_tokenized = Counter()
dictionary_stemmed = Counter()
dictionary_non_tokenized = Counter()

# max chunks to be processed. makes it easier to test
max_chunks = 2

# creates
for i, chunk in enumerate(pd.read_csv("995,000_rows.csv", chunksize=100000)):
    if max_chunks is not None and i >= max_chunks:
        break

    # process single chunk
    chunk = clean.data_pipeline(chunk)
    
    # update dictionaries for processed chunk by adding the dictionaries of chunks together
    dictionary_tokenized.update(
        clean.build_dictionary(chunk["token_without_stopwords"])
    )
    
    dictionary_stemmed.update(
        clean.build_dictionary(chunk["stemmed_text"])
    )
    
    dictionary_non_tokenized.update(
        word for row in chunk["cleaned_text"] for word in row.split()
    )

    chunk[["content", "type", "stemmed_text"]].to_csv(
        "processed_data.csv",
        mode="a",
        index=False,
        header=(i == 0)
    )
    # chunk data is not joined together in memory, but saved on disk as a csv file


print("Dictionary size of tokenized and stopwords removed:", len(dictionary_tokenized))
print("dictionary size of nontokenized and no stopword removal:", len(dictionary_non_tokenized))
print("Size reduction:", len(dictionary_non_tokenized) - len(dictionary_tokenized))
print("Dictionary size after stemming:", len(dictionary_stemmed))
print("Reduction rate after stemming:",
    100 - (len(dictionary_stemmed) / len(dictionary_tokenized) * 100))
clean.plot_most_frequent_words(dictionary_tokenized)

'''
Output when run:
Dictionary size of tokenized and stopwords removed: 2120425
dictionary size of nontokenized and no stopword removal: 2817928
Size reduction: 697503
Dictionary size after stemming: 1862236
Reduction rate after stemming: 12.176285414480589 (a bit low?)
'''
