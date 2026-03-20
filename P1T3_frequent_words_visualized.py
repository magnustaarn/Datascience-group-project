import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import Clean as clean

# Load the saved dictionaries
with open('dictionaries.pkl', 'rb') as f:
    data = pickle.load(f)

dict_tokenized = data['tokenized']
dict_stemmed = data['stemmed']

# top 30 most frequent words, tokenized
plt.figure(figsize=(12, 6))
clean.plot_most_frequent_words_from_dict(dict_tokenized, 30)
plt.title("Top 30 Most Frequent Words (Tokenized)")
plt.yscale('linear')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# top 30 most frequent words, stemmmed
plt.figure(figsize=(12, 6))
clean.plot_most_frequent_words_from_dict(dict_stemmed, 30)
plt.title("Top 30 Most Frequent Words (stemmed)")
plt.yscale('linear')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()