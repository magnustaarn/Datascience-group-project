import re
with open("news_sample.csv", "r", encoding="utf-8") as f:
    data = f.readlines()

def clean_text(data_input):
    # lowercase 
    lines = [line.lower() for line in data_input]
    # remove newline
    lines = [re.sub(r"\n$", "", line) for line in lines]
    # remove tabs
    lines = [re.sub(r"\t", "", line) for line in lines]
    # remove multiple whitespaces
    lines = [re.sub(r"\s{2,}", " ", line) for line in lines]
    # replace email
    lines = [re.sub(r"[0-9a-zA-Z._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", r"<EMAIL>", line) for line in lines]
    # replace url starting with https or www
    lines = [re.sub(r'(?:https?://|www\.)[^\s,"\']+', r"<URL>", line) for line in lines] 
    # replace other url's ending in com, net or org 
    lines = [re.sub(r"[0-9a-zA-Z]+\.(com|net|org)", r"<URL>", line) for line in lines]
    # replace date
    lines = [re.sub(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", r"<DATE>", line) for line in lines]
    # replace numbers
    lines = [re.sub(r"[0-9]+", r"<NUM>", line) for line in lines]
    return lines
cleaned_data = clean_text(data)
print(cleaned_data)