from sklearn.model_selection import train_test_split
import pandas as pd
import os

# delete files if they already exist
for filename in ["train.csv", "validation.csv", "test.csv"]:
    if os.path.exists(filename):
        os.remove(filename)

for i, chunk in enumerate(pd.read_csv("processed_data.csv", chunksize=100000)):
    if len(chunk) < 10:
        continue
    # first split: 80% train, 20% temp
    train_chunk, temp_chunk = train_test_split(
        chunk, test_size=0.2
    )

    # second split: split temp into 10% validation, 10% test
    validation_chunk, test_chunk = train_test_split(
        temp_chunk, test_size=0.5
    )

    train_chunk.to_csv("train.csv", mode="a", index=False, header=(i == 0))
    validation_chunk.to_csv("validation.csv", mode="a", index=False, header=(i == 0))
    test_chunk.to_csv("test.csv", mode="a", index=False, header=(i == 0))
