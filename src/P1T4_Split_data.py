from sklearn.model_selection import train_test_split
import pandas as pd
from paths import DATA_DIR

# delete files if they already exist
for filename in ["train.csv", "validation.csv", "test.csv"]:
    file_path = DATA_DIR / filename
    if file_path.exists():
        file_path.unlink()

input_file = DATA_DIR / "processed_data.csv"
train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"
val_file = DATA_DIR / "validation.csv"

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=100000)):
    if len(chunk) < 10:
        continue
    # first split: 80% train, 20% temp
    train_chunk, temp_chunk = train_test_split(
        chunk,
        test_size=0.2,
        random_state=42
    )

    # second split: split temp into 10% validation, 10% test
    validation_chunk, test_chunk = train_test_split(
        temp_chunk, 
        test_size=0.5,
        random_state=42
    )

    train_chunk.to_csv(train_file, mode="a", index=False, header=(i == 0))
    validation_chunk.to_csv(val_file, mode="a", index=False, header=(i == 0))
    test_chunk.to_csv(test_file, mode="a", index=False, header=(i == 0))
print("saved train data as train.csv\nsaved validation data as validation.csv\nsaved test data as test.csv")
