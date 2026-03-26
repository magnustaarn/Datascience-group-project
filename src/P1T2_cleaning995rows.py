import pandas as pd
import P1T1_Cleaning as clean
from paths import DATA_DIR
from collections import Counter

dictionary_tokenized = Counter()
dictionary_stemmed = Counter()
dictionary_non_tokenized = Counter()

# delete processed_data.csv if it exists
processed_file = DATA_DIR / "processed_data.csv"
if processed_file.exists():
    processed_file.unlink()

input_file = DATA_DIR / "995,000_rows.csv"
output_file = DATA_DIR / "processed_data.csv"
dictionary_file = DATA_DIR / "dictionaries.pkl"
#defines news type as either relaible or not. creates new "label"
def map_label(x):
    if x == "reliable":
        return 1
    elif x in ["fake", "bias", "conspiracy", "satire", "junksci", "hate", "clickbait", "political"]:
        return 0
    return None
# max chunks to be processed.
max_chunks = None

# creates
for i, chunk in enumerate(pd.read_csv(input_file, chunksize=100000)):
    if max_chunks is not None and i >= max_chunks:
        break
    
    # process single chunk
    chunk = clean.data_pipeline(chunk)

    # creates "label"
    chunk["label"] = chunk["type"].apply(map_label)

    # remove rows where label could not be assigned
    chunk = chunk.dropna(subset=["label"])

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

    # convert stemmed text to plain text string
    chunk["stemmed_text"] = chunk["stemmed_text"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )

    columns_to_exclude = [
        "content",
        "cleaned_text",
        "tokenized_text",
        "token_without_stopwords"
    ]
    chunk_to_save = chunk.drop(columns=columns_to_exclude, errors="ignore")

    chunk_to_save.to_csv(
        output_file,
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

import pickle

#Save dictionaries
with open(dictionary_file, 'wb') as f:
    pickle.dump({
        'tokenized': dictionary_tokenized,
        'stemmed': dictionary_stemmed,
        'non_tokenized': dictionary_non_tokenized
    }, f)
print("Dictionaries saved as dictionaries.pkl")

'''
Output when run:
Dictionary size of tokenized and stopwords removed: 2002863
dictionary size of nontokenized and no stopword removal: 2166138
Size reduction: 163275
Dictionary size after stemming: 1765207
Reduction rate after stemming: 11.865814087134268
Dictionaries saved as dictionaries.pkl
'''
