from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def split(df, test_size, n_bins, seed):
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    y_bins = pd.qcut(y, q=n_bins, labels=False)
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y_bins)


def main():
    p = load_params()["split"]
    df = pd.read_csv("data/processed/dataset.csv")
    X_train, X_test, y_train, y_test = split(
        df, p["test_size"], p["n_bins"], p["seed"])
    out = Path("data/split")
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)
    print("Split :", X_train.shape, X_test.shape)


if __name__ == "__main__":
    main()