import pandas as pd
from collections import Counter
import P1T1_Cleaning as clean
from paths import DATA_DIR

dict_tokenized = Counter()
dict_stemmed = Counter()
dict_non_tokenized = Counter()

type_distribution = Counter()
missing_content = 0

# for length of fake and reliable articles - comparison
total_len_reliable = 0
count_reliable = 0
total_len_fake = 0
count_fake = 0

input_file = DATA_DIR / "995,000_rows.csv"
# processing in chunk - faster run time
for chunk in pd.read_csv(input_file, chunksize=100000):
    chunk = clean.data_pipeline(chunk)
    
    # update with data from chunk
    dict_tokenized.update(clean.build_dictionary(chunk["token_without_stopwords"]))
    dict_stemmed.update(clean.build_dictionary(chunk["stemmed_text"]))

    type_distribution.update(chunk["type"].fillna("Unknown")) # occurances of each news type
    missing_content += chunk["content"].isna().sum() # Sums up NaN values in the content column for this chunk - articles without content 

    #Comparing length of reliable and fake news
    chunk['label'] = chunk['type'].apply(clean.map_label) 
    chunk['word_count'] = chunk['content'].fillna("").str.split().str.len() #length of each article

    # reliable
    rel_chunk = chunk[chunk['label'] == 1]
    total_len_reliable += rel_chunk['word_count'].sum()
    count_reliable += len(rel_chunk)

    # fake
    fake_chunk = chunk[chunk['label'] == 0]
    total_len_fake += fake_chunk['word_count'].sum()
    count_fake += len(fake_chunk)

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

# average amount of words in artcile of each type
avg_reliable = total_len_reliable / count_reliable if count_reliable > 0 else 0
avg_fake = total_len_fake / count_fake if count_fake > 0 else 0

# Print dictionaries
print("URL count:", url_count)
print("Date count:", date_count)
print("Number count:", num_count)

print("Distribution of types:", type_distribution)
print("Missing content values:", missing_content)

print(f"Average length of Reliable artciles: {avg_reliable:.2f} words")
print(f"Average length of Fake artciles: {avg_fake:.2f} words")

# Before and after
clean.plot_most_frequent_words_from_dict(dict_non_tokenized, 100)
clean.plot_most_frequent_words_from_dict(dict_stemmed, 100)
clean.plot_most_frequent_words_from_dict(dict_tokenized, 100)