import re
import pandas as pd

df = pd.read_csv("news_sample.csv") # read CSV

def clean_text(text):
    cleaned_text = text.lower() # highercase -> lowercase
    cleaned_text = re.sub(r"\s+", " ", cleaned_text) # line shift, several spaces and tab
    cleaned_text = re.sub(r"\d[\d,\.]*", "<NUM>", cleaned_text) # any number
    cleaned_text = re.sub(r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<DATE>", cleaned_text) # date with time
    cleaned_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", cleaned_text) # date
    cleaned_text = re.sub(r"\S+@\S+", "<EMAIL>", cleaned_text)  # mail
    cleaned_text = re.sub(r"(https?://\S+|www\.\S+)", "<URL>", cleaned_text)  # URL
    cleaned_text = cleaned_text.strip()
    return cleaned_text

df["cleaned_text"] = df["content"].apply(clean_text) # clean specific column
print(df["cleaned_text"]) # print all articles