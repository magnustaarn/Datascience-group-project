# Fake News Data Science Project
This repository contains our exam project on fake news detection. The project uses text and meta data from news articles to train models that classify articles as fake news or not.

# Usage

## Used libraries:
Pandas
nltk
matplotlib
sklearn

use the following command to install the used libraries:
pip install -r requirements.txt

## Running the code
The source code is located in the src directory. Before running the code, the files "995,000.csv", "news_sample.csv" and "liar_dataset.zip" into the "Data" directory. The zip file has to be unzipped inside the folder, such that the tsv files are in the \Data directory. Its important to note that the python files have to be ran in order: "P1T1 -> P1T2 -> P1T3 -> ... -> P4". In addition, the file "P3_neural_network.py" has to be run before "P3_Graph_model_comparison.py".