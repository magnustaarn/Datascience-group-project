import pandas as pd
import Clean as clean

dictionary_tokenized = set()
dictionary_stemmed = set()
dictionary_non_tokenized = set()

# creates
for chunk in pd.read_csv("995,000_rows.csv", chunksize=100000):
    
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
        set(word for row in chunk["cleaned_text"] for word in row.split())
    )
    # chunk data is not joined together for now, only dictionaries are


print("Dictionary size of tokenized and stopwords removed:", len(dictionary_tokenized))
print("dictionary size of nontokenized and no stopword removal:", len(dictionary_non_tokenized))
print("Size reduction:", len(dictionary_non_tokenized) - len(dictionary_tokenized))
print("Dictionary size after stemming:", len(dictionary_stemmed))
print("Reduction rate after stemming:",
    100 - (len(dictionary_stemmed) / len(dictionary_tokenized) * 100))

'''
Output when run:
Dictionary size of tokenized and stopwords removed: 2120425
dictionary size of nontokenized and no stopword removal: 2817928
Size reduction: 697503
Dictionary size after stemming: 1862236
Reduction rate after stemming: 12.176285414480589 (a bit low?)
'''
