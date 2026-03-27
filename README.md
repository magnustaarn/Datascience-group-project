# Fake News Data Science Project
This repository contains our exam project on fake news detection. The project uses text and meta data from news articles to train models that classify articles as fake news or not.

## Used libraries:
Pandas

nltk

matplotlib

sklearn

use the following command to install the used libraries:
pip install -r requirements.txt

## Running the code
The source code is located in the src directory. Before running the code, the files "995,000_rows.csv", "news_sample.csv" and "liar_dataset.zip" into the "Data" directory. The zip file has to be unzipped inside the folder, such that the tsv files are in the \Data directory. It is important to note that the Python files must be run in order: "P1T1 -> P1T2 -> P1T3 -> ... -> P4". In addition, the file "P3_neural_network.py" has to be run before "P3_Graph_model_comparison.py".
