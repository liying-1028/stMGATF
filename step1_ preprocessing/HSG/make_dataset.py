import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("labels.csv")

    ids = np.asarray(df["ID"])
    labels = np.asarray(df["Cluster"])
    print(ids.shape)
    print(labels.shape, labels.dtype)

    train_ids, val_ids, train_labels, val_labels = train_test_split(ids, labels, test_size=0.2)

    for idx, label in zip(train_ids, train_labels):
        Path(f"data/dna/train/{label:02d}").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            f"images/1/{idx}40.jpeg",
            f"data/dna/train/{label:02d}/{idx}40.jpeg",
        )

    for idx, label in zip(val_ids, val_labels):
        Path(f"data/dna/val/{label:02d}").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            f"images/1/{idx}40.jpeg",
            f"data/dna/val/{label:02d}/{idx}40.jpeg",
        )


if __name__ == "__main__":
    main()
