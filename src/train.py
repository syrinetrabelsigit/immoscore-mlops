from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression


def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def main():
    X_train = pd.read_csv("data/split/X_train.csv")
    y_train = pd.read_csv("data/split/y_train.csv").squeeze("columns")
    model = train(X_train, y_train)
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modèle sauvegardé")


if __name__ == "__main__":
    main()