import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import Clean as clean

#Counting
url_count = dictionary_non_tokenized["URL_TOKEN"]
date_count = dictionary_non_tokenized["DATE_TOKEN"]
num_count = dictionary_non_tokenized["NUM_TOKEN"]

#Frequency of words
dictionary_tokenized.most_common(100)

top100words = dictionary_tokenized.most_common(100)
for word, count in top100words:
  print(word, count)

clean.plot_most_frequent_words(dictionary_tokenized)
