from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def process(df, corr_threshold):
    df_num = df.select_dtypes(include=[np.number]).dropna()
    corr = df_num.corr()["SalePrice"].abs()
    cols = corr[corr > corr_threshold].index.tolist()
    return df_num[cols]


def main():
    params = load_params()
    df = pd.read_csv("data/raw/data.csv")
    out = process(df, params["process"]["corr_threshold"])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/processed/dataset.csv", index=False)
    print("Dataset préparé :", out.shape)


if __name__ == "__main__":
    main()