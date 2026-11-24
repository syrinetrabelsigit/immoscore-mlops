from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def metrics(model, X, y):
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def main():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    res = {}
    for name in ["train", "test"]:
        X = pd.read_csv(f"data/split/X_{name}.csv")
        y = pd.read_csv(f"data/split/y_{name}.csv").squeeze("columns")
        res[name] = metrics(model, X, y)
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()