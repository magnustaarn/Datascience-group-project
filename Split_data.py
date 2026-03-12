from sklearn.model_selection import train_test_split
import pandas as pd

for i, chunk in enumerate(pd.read_csv("processed_data.csv", chunksize=100000)):
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
