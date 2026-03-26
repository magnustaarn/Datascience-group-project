import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import P1T1_Cleaning as clean
from paths import DATA_DIR

dictionary_file = DATA_DIR / "dictionaries.pkl"
# Load the saved dictionaries
with open(dictionary_file, 'rb') as f:
    data = pickle.load(f)

dict_tokenized = data['tokenized']
dict_stemmed = data['stemmed']

# top 30 most frequent words, tokenized
plt.figure(figsize=(12, 6))
clean.plot_most_frequent_words_from_dict(dict_tokenized, 30)
plt.title("Top 30 Most Frequent Words (Tokenized)")
plt.yscale('linear')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig("top_30_tokenized.png", bbox_inches='tight') # saving file in dict
print("Saved top_30_tokenized.png")

# top 30 most frequent words, stemmmed
plt.figure(figsize=(12, 6))
clean.plot_most_frequent_words_from_dict(dict_stemmed, 30)
plt.title("Top 30 Most Frequent Words (stemmed)")
plt.yscale('linear')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig("top_30_stemmed.png", bbox_inches='tight') # saving file in dict
print("Saved top_30_stemmed.png")

plt.show()